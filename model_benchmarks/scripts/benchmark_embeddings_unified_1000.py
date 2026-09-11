"""
5개 임베딩 모델을 FAQ 검색 정확도 + 성능 지표로 자동 벤치마크하는 스크립트.
(서빙 방식 통일 버전: Ollama를 거치지 않고 5개 모델 전부 sentence-transformers로 직접 로딩합니다.
 모델 자체의 정확도/속도를 순수하게 비교하기 위한 버전이며, 실제 Ollama 배포 속도와는 다를 수 있습니다.)

EmbeddingTestFAQ_통합.xlsx(FAQ 1,000개 + 의도 분류 포함 최신 스키마)를 기준으로 동작합니다.
FAQ 코퍼스는 건별이 아닌 배치로 임베딩해서, 1,000개 규모에서도 속도 저하를 최소화합니다.

측정 지표:
  정확도 - Top-1 / Top-3 / Top-5 / MRR / Hard Negative 정확도
          (+ 카테고리별: Easy / Medium / Hard / 구어체 / 오타 / 긴질문)
  성능   - 평균 임베딩 시간 / P95 임베딩 시간 / 초당 처리량(단일 요청 기준) / FAQ 코퍼스 전체 임베딩 시간
  리소스 - RAM(프로세스 메모리 증가분) / VRAM(GPU 사용 시 자동 측정)

사전 준비:
  pip install sentence-transformers torch openpyxl psutil

  최초 실행 시 각 모델이 HuggingFace에서 자동으로 다운로드됩니다 (수백MB~1GB대,
  모델마다 다름). 미리 받아둘 필요는 없고, 스크립트가 알아서 캐시에 저장합니다.

⚠️ RAM 측정 관련 주의사항:
  5개 모델을 한 프로세스 안에서 순서대로 로딩하기 때문에, 이전 모델의 메모리가
  완전히 회수되지 않은 채로 다음 모델의 RAM이 측정될 수 있습니다. 정확도/속도
  지표는 모델별로 독립적이라 상관없지만, RAM만큼은 --models 옵션으로 모델을
  하나씩 별도 프로세스로 실행해서 측정하는 것을 권장합니다.

  gte-multilingual-base 등 일부 모델은 저장소에 포함된 커스텀 코드로 동작하여
  trust_remote_code=True가 필요합니다. 이 스크립트는 모든 모델에 이 옵션을
  적용하므로, MODEL_CONFIGS에 신뢰할 수 없는 출처의 모델을 추가하지 마세요.

사용법:
  python benchmark_embeddings_unified.py --xlsx EmbeddingTestFAQ_통합.xlsx
  (별도로 --output을 지정하지 않으면 benchmark_results_unified_summary.csv / _detail.csv로 저장되어
   기존 benchmark_embeddings.py의 결과 파일과 겹치지 않습니다.)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import subprocess
import sys
import time

import openpyxl
import requests

OLLAMA_URL = "http://localhost:11434/api/embed"

# ── 모델 설정: 5개 전부 sentence-transformers로 직접 로딩 (서빙 방식 통일) ──────
# prompt_style 종류:
#   "none"        - 접두사/프롬프트 없이 그대로 인코딩 (bge-m3, gte-multilingual-base)
#   "prefix"      - 텍스트 앞에 문자열을 붙임 (multilingual-e5-large: "query: "/"passage: ")
#   "prompt_name" - sentence-transformers의 내장 prompt_name 파라미터 사용 (qwen3-embedding)
#   "method"      - 모델 전용 encode_query()/encode_document() 메서드 사용 (embeddinggemma)
MODEL_CONFIGS = [
    {
        "name": "qwen3-embedding:0.6b",
        "backend": "sentence_transformers",
        "hf_id": "Qwen/Qwen3-Embedding-0.6B",
        "prompt_style": "prompt_name",
    },
    {
        "name": "bge-m3",
        "backend": "sentence_transformers",
        "hf_id": "BAAI/bge-m3",
        "prompt_style": "none",
    },
    {
        "name": "multilingual-e5-large",
        "backend": "sentence_transformers",
        "hf_id": "intfloat/multilingual-e5-large",
        "prompt_style": "prefix",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    {
        "name": "embeddinggemma",
        "backend": "sentence_transformers",
        "hf_id": "google/embeddinggemma-300m",
        "prompt_style": "method",
    },
    {
        "name": "gte-multilingual-base",
        "backend": "sentence_transformers",
        "hf_id": "Alibaba-NLP/gte-multilingual-base",
        "prompt_style": "none",
    },
]

# 참고: Ollama로 서빙하고 싶은 모델이 나중에 생기면, backend를 "ollama"로 바꾸고
# "ollama_tag"를 지정하면 됩니다 (OllamaEmbedder 클래스는 아래에 그대로 남아있습니다).

# 테스트 쿼리가 들어있는 시트와, 결과에 표시할 카테고리 라벨 매핑
QUERY_SHEETS = {
    "Retrieval Easy": "Easy",
    "Retrieval Medium": "Medium",
    "Retrieval Hard": "Hard",
    "Retrieval 구어체": "구어체",
    "Retrieval 오타": "오타",
    "Retrieval 긴질문": "긴질문",
    "Retrieval 확장": "확장",
}


# ────────────────────────────── 데이터 로딩 ──────────────────────────────

def split_ids(cell_value) -> list[str]:
    """'FAQ-051; FAQ-010' 또는 'FAQ-051, FAQ-010'처럼 세미콜론/쉼표 어느 쪽으로
    구분되어 있어도 모두 인식해서 분리한다 (원본 시트마다 구분자가 다를 수 있음)."""
    if not cell_value:
        return []
    import re

    return [x.strip() for x in re.split(r"[;,]", str(cell_value)) if x.strip()]


def load_faq_corpus(xlsx_path: str) -> dict[str, str]:
    """FAQ 시트를 읽어서 {FAQ ID: 질문 텍스트} 딕셔너리로 반환한다.
    (헤더가 1행에 있는 최신 스키마 기준: FAQ ID, 카테고리, FAQ 질문, FAQ 답변, 권장 처리 의도, 세부 의도)"""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb["FAQ"]

    corpus = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        faq_id, category, question = row[0], row[1], row[2]
        if faq_id:
            corpus[faq_id] = question
    return corpus


def load_test_queries(xlsx_path: str) -> list[dict]:
    """
    6개 Retrieval 시트를 읽어서 FAQ_RAG 의도인 쿼리만 반환한다.
    시트마다 컬럼 구성이 조금씩 달라서(Easy/Medium/구어체/오타: 7열, Hard/긴질문: 8열),
    헤더 이름으로 컬럼 위치를 찾아 유연하게 처리한다.
    """
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)

    queries = []
    for sheet_name, category_label in QUERY_SHEETS.items():
        if sheet_name not in wb.sheetnames:
            print(f"[경고] 시트를 찾을 수 없어 건너뜀: {sheet_name}")
            continue

        ws = wb[sheet_name]
        headers = [c.value for c in ws[1]]

        def col_idx(*keywords):
            for i, h in enumerate(headers):
                if h and any(k in h for k in keywords):
                    return i
            return None

        idx_id = 0
        idx_query = 1
        idx_gt = col_idx("Primary GT")
        idx_acceptable = col_idx("Acceptable GT")
        idx_hardneg = col_idx("Hard Negative", "경쟁 FAQ", "경쟁")
        idx_intent = col_idx("처리 의도")

        for row in ws.iter_rows(min_row=2, values_only=True):
            test_id, user_query = row[idx_id], row[idx_query]
            if not test_id or not user_query:
                continue

            intent = row[idx_intent] if idx_intent is not None else "FAQ_RAG"
            if intent and intent != "FAQ_RAG":
                continue  # MAP_API 등 FAQ 검색과 무관한 쿼리는 제외

            gt_ids = split_ids(row[idx_gt]) if idx_gt is not None else []
            acceptable_ids = split_ids(row[idx_acceptable]) if idx_acceptable is not None else []
            hard_neg_ids = split_ids(row[idx_hardneg]) if idx_hardneg is not None else []

            queries.append(
                {
                    "test_id": test_id,
                    "category": category_label,
                    "query": user_query,
                    "gt_ids": gt_ids,
                    "acceptable_ids": acceptable_ids or gt_ids,
                    "hard_neg_ids": hard_neg_ids,
                }
            )
    return queries


# ────────────────────────────── 임베딩 백엔드 ──────────────────────────────

class OllamaEmbedder:
    def __init__(self, ollama_tag: str, query_prefix: str = "", passage_prefix: str = ""):
        self.tag = ollama_tag
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def embed(self, text: str, mode: str = "query") -> tuple[list[float], float]:
        """텍스트를 임베딩하고 (벡터, 클라이언트 소요시간초)를 반환한다.

        mode: "query"면 사용자 질문용 접두사를, "passage"면 문서(FAQ)용 접두사를 붙인다.
        접두사가 설정되지 않은 모델은 그대로 원문을 보낸다.
        """
        prefix = self.query_prefix if mode == "query" else self.passage_prefix
        full_text = f"{prefix}{text}"

        payload = {"model": self.tag, "input": full_text}
        start = time.perf_counter()
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            print("[오류] Ollama 서버에 연결할 수 없습니다. 'ollama serve' 실행 여부를 확인하세요.")
            sys.exit(1)
        except requests.exceptions.HTTPError as e:
            print(f"[오류] '{self.tag}' 호출 실패: {e} / 응답: {resp.text}")
            print(f"모델이 pull 되어 있는지 확인하세요: ollama pull {self.tag}")
            sys.exit(1)
        elapsed = time.perf_counter() - start

        data = resp.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            print(f"[오류] 응답에 embeddings가 없습니다: {data}")
            sys.exit(1)
        return embeddings[0], elapsed

    def memory_usage(self) -> dict:
        """'ollama ps' 출력에서 현재 모델의 메모리 사용량(SIZE 컬럼)을 파싱한다."""
        try:
            out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10)
        except FileNotFoundError:
            return {"note": "ollama 명령을 찾을 수 없습니다. PATH를 확인하세요."}

        lines = out.stdout.strip().splitlines()
        if len(lines) < 2:
            return {"note": "로드된 모델이 없습니다. embed 호출 직후에 다시 시도하세요."}

        header = lines[0].split()
        size_idx = header.index("SIZE") if "SIZE" in header else None

        for line in lines[1:]:
            if self.tag.split(":")[0] in line:
                parts = line.split()
                size_val = parts[size_idx] if size_idx is not None and size_idx < len(parts) else "?"
                return {"raw_line": line.strip(), "size": size_val}

        return {"note": f"'{self.tag}'가 ollama ps 목록에 없습니다. 방금 embed를 호출했는지 확인하세요."}


class SentenceTransformerEmbedder:
    def __init__(
        self,
        hf_id: str,
        prompt_style: str = "none",
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        try:
            import psutil
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("[오류] sentence-transformers / psutil이 설치되어 있지 않습니다.")
            print("       pip install sentence-transformers torch psutil")
            sys.exit(1)

        self.prompt_style = prompt_style
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

        self._psutil = psutil
        process = psutil.Process()
        rss_before = process.memory_info().rss

        # GPU(CUDA)를 명시적으로 감지해서 사용. 없으면 CPU로 자동 전환.
        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
        except ImportError:
            pass

        print(f"'{hf_id}' 로딩 중... (device={device}, 처음 실행 시 다운로드 때문에 오래 걸릴 수 있습니다)")
        if device == "cpu":
            print("  ⚠️ GPU가 감지되지 않아 CPU로 로딩합니다. GPU를 쓰려면 CUDA 지원 torch가 설치되어 있어야 합니다.")
        # gte-multilingual-base 등 일부 모델은 저장소에 포함된 커스텀 코드로 구현되어 있어
        # trust_remote_code=True가 필요합니다. 신뢰할 수 있는 출처의 모델에서만 사용하세요.
        self.model = SentenceTransformer(hf_id, device=device, trust_remote_code=True)
        self.device = device

        rss_after = process.memory_info().rss
        self._ram_delta_mb = (rss_after - rss_before) / (1024 * 1024)

        self._vram_mb = None
        try:
            import torch

            if torch.cuda.is_available():
                self._vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        except ImportError:
            pass

    def embed(self, text: str, mode: str = "query") -> tuple[list[float], float]:
        """prompt_style에 따라 query/passage를 다르게 인코딩한다.

        - "none": 접두사/프롬프트 없이 그대로
        - "prefix": query_prefix/passage_prefix 문자열을 앞에 붙임 (예: e5 계열)
        - "prompt_name": sentence-transformers 내장 prompt_name="query" 사용, 문서는 프롬프트 없음
        - "method": 모델 전용 encode_query()/encode_document() 메서드 사용 (예: embeddinggemma)
        """
        start = time.perf_counter()

        if self.prompt_style == "prefix":
            prefix = self.query_prefix if mode == "query" else self.passage_prefix
            vector = self.model.encode(f"{prefix}{text}", normalize_embeddings=True)

        elif self.prompt_style == "prompt_name":
            prompt_name = "query" if mode == "query" else None
            vector = self.model.encode(text, prompt_name=prompt_name, normalize_embeddings=True)

        elif self.prompt_style == "method":
            if mode == "query":
                vector = self.model.encode_query(text)
            else:
                vector = self.model.encode_document(text)

        else:  # "none"
            vector = self.model.encode(text, normalize_embeddings=True)

        elapsed = time.perf_counter() - start
        return vector.tolist(), elapsed

    def embed_batch(self, texts: list[str], mode: str = "passage"):
        """다수의 텍스트(예: FAQ 코퍼스)를 한 번에 배치로 임베딩한다.
        embed()와 동일한 prompt_style 규칙을 따르되, 건별 호출 대신 배치 처리로 속도를 높인다."""
        if self.prompt_style == "prefix":
            prefix = self.query_prefix if mode == "query" else self.passage_prefix
            texts = [f"{prefix}{t}" for t in texts]
            return self.model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)

        elif self.prompt_style == "prompt_name":
            prompt_name = "query" if mode == "query" else None
            return self.model.encode(
                texts, prompt_name=prompt_name, normalize_embeddings=True, batch_size=64, show_progress_bar=False
            )

        elif self.prompt_style == "method":
            if mode == "query":
                return self.model.encode_query(texts)
            return self.model.encode_document(texts)

        else:  # "none"
            return self.model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)

    def memory_usage(self) -> dict:
        return {
            "device": self.device,
            "ram_delta_mb": round(self._ram_delta_mb, 1),
            "vram_mb": round(self._vram_mb, 1) if self._vram_mb is not None else "N/A (GPU 미사용 또는 CPU 모드)",
        }


def build_embedder(config: dict):
    if config["backend"] == "ollama":
        return OllamaEmbedder(
            config["ollama_tag"],
            query_prefix=config.get("query_prefix", ""),
            passage_prefix=config.get("passage_prefix", ""),
        )
    elif config["backend"] == "sentence_transformers":
        return SentenceTransformerEmbedder(
            config["hf_id"],
            prompt_style=config.get("prompt_style", "none"),
            query_prefix=config.get("query_prefix", ""),
            passage_prefix=config.get("passage_prefix", ""),
        )
    raise ValueError(f"알 수 없는 backend: {config['backend']}")


# ────────────────────────────── 유사도 / 랭킹 ──────────────────────────────

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_faqs(query_vec: list[float], faq_vectors: dict[str, list[float]]) -> list[str]:
    """FAQ ID를 유사도 내림차순으로 정렬해서 반환한다."""
    scored = [(faq_id, cosine_similarity(query_vec, vec)) for faq_id, vec in faq_vectors.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [faq_id for faq_id, _ in scored]


def first_rank(ranked_ids: list[str], target_ids: list[str]) -> int | None:
    """target_ids 중 가장 먼저(높은 순위) 등장하는 위치(1부터 시작)를 반환. 없으면 None."""
    for pos, faq_id in enumerate(ranked_ids, start=1):
        if faq_id in target_ids:
            return pos
    return None


# ────────────────────────────── 모델 1개 평가 ──────────────────────────────

def evaluate_model(config: dict, faq_corpus: dict[str, str], queries: list[dict]) -> dict:
    name = config["name"]
    print(f"\n{'=' * 60}\n모델 평가 시작: {name}\n{'=' * 60}")

    embedder = build_embedder(config)

    # 1) FAQ 코퍼스 임베딩 (배치 처리 - FAQ가 1,000개 규모라 건별 호출보다 훨씬 빠름)
    print(f"FAQ {len(faq_corpus)}개 임베딩 중...")
    faq_ids = list(faq_corpus.keys())
    faq_texts = list(faq_corpus.values())

    faq_embed_start = time.perf_counter()
    faq_vecs = embedder.embed_batch(faq_texts, mode="passage")
    faq_embed_time = time.perf_counter() - faq_embed_start
    print(f"FAQ {len(faq_corpus)}개 임베딩 완료: {faq_embed_time:.2f}초")

    faq_vectors = dict(zip(faq_ids, [v.tolist() if hasattr(v, "tolist") else v for v in faq_vecs]))

    # 2) 리소스 사용량 (FAQ 임베딩 직후, 모델이 메모리에 로드된 상태에서 측정)
    memory_info = embedder.memory_usage()

    # 3) 쿼리별 평가
    per_query_results = []
    latencies = []

    print(f"테스트 쿼리 {len(queries)}개 평가 중...")
    for i, q in enumerate(queries, start=1):
        query_vec, elapsed = embedder.embed(q["query"], mode="query")
        latencies.append(elapsed)

        ranked = rank_faqs(query_vec, faq_vectors)

        gt_rank = first_rank(ranked, q["gt_ids"])
        acceptable_rank = first_rank(ranked, q["acceptable_ids"])
        hard_neg_rank = first_rank(ranked, q["hard_neg_ids"]) if q["hard_neg_ids"] else None

        top1 = gt_rank == 1
        top3 = gt_rank is not None and gt_rank <= 3
        top5 = gt_rank is not None and gt_rank <= 5
        rr = (1 / gt_rank) if gt_rank else 0.0

        # Hard Negative 통과: GT가 모든 Hard Negative보다 높은 순위(더 낮은 숫자)에 있어야 함
        hard_neg_pass = True
        if q["hard_neg_ids"]:
            hard_neg_pass = (gt_rank is not None) and (hard_neg_rank is None or gt_rank < hard_neg_rank)

        per_query_results.append(
            {
                "model": name,
                "test_id": q["test_id"],
                "category": q["category"],
                "gt_rank": gt_rank,
                "acceptable_rank": acceptable_rank,
                "top1": top1,
                "top3": top3,
                "top5": top5,
                "rr": rr,
                "hard_neg_pass": hard_neg_pass,
                "latency_sec": elapsed,
            }
        )

        if i % 10 == 0:
            print(f"  {i}/{len(queries)} 완료")

    # 4) 집계 (전체 + 카테고리별)
    summary = summarize(name, per_query_results, latencies, memory_info, faq_embed_time)

    # 5) 다음 모델의 RAM 측정이 이 모델의 잔여 메모리에 오염되지 않도록 정리 시도.
    #    (같은 프로세스 안에서 여러 모델을 순서대로 로딩하기 때문에 완벽하지는 않음 -
    #     RAM 수치를 확실히 믿으려면 --models 옵션으로 모델을 하나씩 별도 실행하는 것을 권장)
    del embedder
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return {"summary": summary, "per_query": per_query_results}


def summarize(model_name: str, results: list[dict], latencies: list[float], memory_info: dict, faq_embed_time: float = None) -> list[dict]:
    """전체 + 카테고리별 지표를 계산한다. results[0]은 '전체', 이후는 카테고리별."""
    categories = ["전체"] + sorted({r["category"] for r in results})

    # 콜드스타트(첫 호출)는 시간 통계에서 제외
    warm_latencies = latencies[1:] if len(latencies) > 1 else latencies
    avg_latency = statistics.mean(warm_latencies) if warm_latencies else 0.0
    sorted_lat = sorted(warm_latencies)
    p95_idx = max(0, math.ceil(0.95 * len(sorted_lat)) - 1)
    p95_latency = sorted_lat[p95_idx] if sorted_lat else 0.0
    throughput = 1 / avg_latency if avg_latency > 0 else 0.0

    rows = []
    for cat in categories:
        subset = results if cat == "전체" else [r for r in results if r["category"] == cat]
        if not subset:
            continue

        n = len(subset)
        top1 = sum(r["top1"] for r in subset) / n
        top3 = sum(r["top3"] for r in subset) / n
        top5 = sum(r["top5"] for r in subset) / n
        mrr = sum(r["rr"] for r in subset) / n
        hard_neg = sum(r["hard_neg_pass"] for r in subset) / n

        row = {
            "model": model_name,
            "category": cat,
            "n": n,
            "top1": round(top1, 3),
            "top3": round(top3, 3),
            "top5": round(top5, 3),
            "mrr": round(mrr, 3),
            "hard_negative_acc": round(hard_neg, 3),
        }
        if cat == "전체":
            row["avg_latency_sec"] = round(avg_latency, 3)
            row["p95_latency_sec"] = round(p95_latency, 3)
            row["throughput_per_sec"] = round(throughput, 2)
            row["memory_info"] = str(memory_info)
            if faq_embed_time is not None:
                row["faq_corpus_embed_time_sec"] = round(faq_embed_time, 3)
        rows.append(row)

    return rows


# ────────────────────────────── 결과 저장 / 출력 ──────────────────────────────

def print_summary_table(all_summaries: list[list[dict]]) -> None:
    print(f"\n{'=' * 100}")
    print("전체 요약 (카테고리 = 전체)")
    print(f"{'=' * 100}")
    header = f"{'모델':<28}{'Top-1':>8}{'Top-3':>8}{'Top-5':>8}{'MRR':>8}{'HardNeg':>10}{'평균(s)':>10}{'P95(s)':>10}{'처리량/s':>10}"
    print(header)
    for summary in all_summaries:
        row = next(r for r in summary if r["category"] == "전체")
        print(
            f"{row['model']:<28}{row['top1']:>8}{row['top3']:>8}{row['top5']:>8}{row['mrr']:>8}"
            f"{row['hard_negative_acc']:>10}{row['avg_latency_sec']:>10}{row['p95_latency_sec']:>10}"
            f"{row['throughput_per_sec']:>10}"
        )

    print(f"\n{'=' * 100}")
    print("카테고리별 Top-1 정확도")
    print(f"{'=' * 100}")
    categories = sorted({r["category"] for s in all_summaries for r in s if r["category"] != "전체"})
    header2 = f"{'모델':<28}" + "".join(f"{c:>12}" for c in categories)
    print(header2)
    for summary in all_summaries:
        by_cat = {r["category"]: r["top1"] for r in summary}
        line = f"{summary[0]['model']:<28}" + "".join(f"{by_cat.get(c, '-'):>12}" for c in categories)
        print(line)


def save_csv(all_summaries: list[list[dict]], all_per_query: list[list[dict]], output_prefix: str) -> None:
    """일괄 저장용 헬퍼. main()은 이제 append_summary_csv/append_detail_csv로 모델별 즉시 저장을 사용하지만,
    별도 스크립트에서 한꺼번에 저장하고 싶을 때 쓸 수 있도록 남겨둔다."""
    for i, (summary, per_query) in enumerate(zip(all_summaries, all_per_query)):
        append_summary_csv(f"{output_prefix}_summary.csv", summary, write_header=(i == 0))
        append_detail_csv(f"{output_prefix}_detail.csv", per_query, write_header=(i == 0))
    print(f"\n결과 저장 완료: {output_prefix}_summary.csv / {output_prefix}_detail.csv")


def append_summary_csv(path: str, summary: list[dict], write_header: bool) -> None:
    fieldnames = [
        "model", "category", "n", "top1", "top3", "top5", "mrr", "hard_negative_acc",
        "avg_latency_sec", "p95_latency_sec", "throughput_per_sec", "memory_info",
        "faq_corpus_embed_time_sec",
    ]

    if os.path.exists(path) and not write_header:
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            existing_header = reader.fieldnames or []
            existing_rows = list(reader)
        if existing_header != fieldnames:
            merged = list(existing_header)
            for name in fieldnames:
                if name not in merged:
                    merged.append(name)
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=merged)
                writer.writeheader()
                for old_row in existing_rows:
                    writer.writerow({k: old_row.get(k, "") for k in merged})
                for row in summary:
                    writer.writerow({k: row.get(k, "") for k in merged})
            print(f"[안내] '{path}'의 컬럼 구성이 이전 실행과 달라 컬럼을 통합해 재정렬했습니다.")
            return

    mode = "w" if write_header else "a"
    with open(path, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in summary:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def append_detail_csv(path: str, per_query: list[dict], write_header: bool) -> None:
    fieldnames = [
        "model", "test_id", "category", "gt_rank", "acceptable_rank",
        "top1", "top3", "top5", "rr", "hard_neg_pass", "latency_sec",
    ]
    mode = "w" if write_header else "a"
    with open(path, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(per_query)


# ────────────────────────────── 메인 ──────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="여러 임베딩 모델을 FAQ 검색 정확도/성능 기준으로 벤치마크합니다.")
    parser.add_argument("--xlsx", required=True, help="FAQ + 테스트 쿼리가 담긴 xlsx 파일 경로")
    parser.add_argument("--output", default="benchmark_results_unified", help="결과 CSV 파일 접두사")
    parser.add_argument(
        "--models",
        default=None,
        help="쉼표로 구분한 모델 이름만 선택 실행 (생략 시 MODEL_CONFIGS 전체 실행)",
    )
    args = parser.parse_args()

    faq_corpus = load_faq_corpus(args.xlsx)
    queries = load_test_queries(args.xlsx)
    print(f"FAQ {len(faq_corpus)}개, 테스트 쿼리 {len(queries)}개를 불러왔습니다.")

    configs = MODEL_CONFIGS
    if args.models:
        selected = {m.strip() for m in args.models.split(",")}
        configs = [c for c in MODEL_CONFIGS if c["name"] in selected]

    summary_path = f"{args.output}_summary.csv"
    detail_path = f"{args.output}_detail.csv"

    # 파일이 이미 존재하면(이전 실행에서 만들어진 것) 헤더를 다시 쓰지 않고 이어서 append한다.
    # 이렇게 해야 --models로 모델을 하나씩 나눠 실행해도 결과가 누적된다.
    write_header = not os.path.exists(summary_path)

    all_summaries = []
    succeeded, failed = [], []

    for config in configs:
        try:
            result = evaluate_model(config, faq_corpus, queries)
        except Exception as e:
            print(f"\n[모델 실패] '{config['name']}' 평가 중 오류 발생, 건너뛰고 다음 모델로 진행합니다.")
            print(f"  오류 내용: {e}")
            failed.append(config["name"])
            continue

        # 모델 하나가 끝나는 즉시 CSV에 기록 (다음 모델이 실패해도 이 결과는 안전하게 남음)
        try:
            append_summary_csv(summary_path, result["summary"], write_header=write_header)
            append_detail_csv(detail_path, result["per_query"], write_header=write_header)
            write_header = False  # 이번 실행에서 한 번이라도 썼으면 그다음부터는 계속 append
            print(f"'{config['name']}' 결과를 {summary_path} / {detail_path}에 저장했습니다.")
        except PermissionError:
            # CSV가 엑셀 등 다른 프로그램에서 열려있어 잠긴 경우. 이미 계산은 끝났으니
            # 결과를 잃지 않도록 콘솔에 출력하고, 타임스탬프 붙은 백업 파일에 별도 저장한다.
            print(f"\n[경고] '{summary_path}' 파일이 잠겨있어 쓸 수 없습니다 (엑셀 등에서 열려있는지 확인하세요).")
            print(f"'{config['name']}' 계산은 끝났으니 결과를 잃지 않기 위해 백업 파일에 저장을 시도합니다.")
            backup_prefix = f"{args.output}_backup_{int(time.time())}"
            try:
                append_summary_csv(f"{backup_prefix}_summary.csv", result["summary"], write_header=True)
                append_detail_csv(f"{backup_prefix}_detail.csv", result["per_query"], write_header=True)
                print(f"백업 저장 완료: {backup_prefix}_summary.csv / {backup_prefix}_detail.csv")
                print(f"원래 파일을 닫은 뒤, 이 백업 파일 내용을 {summary_path}에 수동으로 합쳐주세요.")
            except Exception:
                print("백업 저장도 실패했습니다. 아래 결과를 콘솔에서 직접 복사해두세요:")
                print(result["summary"])

        all_summaries.append(result["summary"])
        succeeded.append(config["name"])

    if all_summaries:
        print_summary_table(all_summaries)

    print(f"\n성공: {succeeded}")
    if failed:
        print(f"실패: {failed} → 문제를 고친 뒤 아래처럼 실패한 모델만 다시 실행하면, 기존 결과 뒤에 이어서 저장됩니다.")
        print(f"  python {sys.argv[0]} --xlsx {args.xlsx} --output {args.output} --models {','.join(failed)}")


if __name__ == "__main__":
    main()

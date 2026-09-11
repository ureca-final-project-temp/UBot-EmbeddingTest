"""
BGE-M3 임베딩 단독 검색 vs BGE-M3 + reranker(2단계) 검색을 같은 60개 테스트 쿼리로 비교하는 스크립트.

방식:
  1단계) BGE-M3로 FAQ 100개 전체와 코사인 유사도를 계산해 Top-K(기본 10개) 후보를 뽑는다.
         -> 이 순위 그대로의 성능이 "임베딩 단독" 결과.
  2단계) Top-K 후보만 reranker(cross-encoder)로 다시 정밀 채점해서 순서를 바꾼다.
         -> 바뀐 순위의 성능이 "임베딩+reranker" 결과.

  주의: reranker는 1단계가 애초에 놓친(Top-K 밖으로 빠진) 정답은 절대 되살릴 수 없다.
       즉 reranker의 효과는 "Top-K 안에서 순서를 더 정확하게 재배열하는 것"으로 제한된다.

  MAP_API 의도로 표시된 쿼리(예: E09, M09, H09 - 지도 라우팅 테스트)는 FAQ 검색 비교와
  무관하므로 자동으로 제외하고 FAQ_RAG 의도 쿼리만 사용한다.

사전 준비:
  pip install sentence-transformers torch scikit-learn openpyxl

사용법:
  python compare_embedding_reranker.py --xlsx EmbeddingTestFAQ_통합.xlsx
  (기본으로 top_k=3,5,10,15,20 전부를 한 번에 테스트합니다. FAQ/쿼리 임베딩은 top_k와 무관하게
   단 한 번만 계산하고, top_k별로는 reranker 단계만 다시 수행해서 효율적으로 비교합니다.)

  python compare_embedding_reranker.py --xlsx EmbeddingTestFAQ_통합.xlsx --rerankers bge-reranker-v2-m3
  python compare_embedding_reranker.py --xlsx EmbeddingTestFAQ_통합.xlsx --top-k 5
  python compare_embedding_reranker.py --xlsx EmbeddingTestFAQ_통합.xlsx --top-k 3,7,15
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

# ── reranker 후보 설정 ────────────────────────────────────────────────
RERANKER_CONFIGS = [
    {"name": "bge-reranker-v2-m3", "hf_id": "BAAI/bge-reranker-v2-m3"},
    {"name": "jina-reranker-v2", "hf_id": "jinaai/jina-reranker-v2-base-multilingual", "trust_remote_code": True},
]

# 검색 결과와 정답을 비교할 쿼리 시트들과, 결과에 표시할 카테고리 라벨
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
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb["FAQ"]
    corpus = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0]:
            corpus[row[0]] = row[2]  # FAQ ID -> FAQ 질문
    return corpus


def load_test_queries(xlsx_path: str) -> list[dict]:
    """
    6개 Retrieval 시트를 읽어서 FAQ_RAG 의도인 쿼리만 반환한다.
    시트마다 컬럼 구성이 조금씩 달라서(Easy/Medium/구어체/오타: 7열, Hard/긴질문: 8열),
    헤더 이름으로 컬럼 위치를 찾아 유연하게 처리한다.
    """
    import openpyxl

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
            test_id, query = row[idx_id], row[idx_query]
            if not test_id or not query:
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
                    "query": query,
                    "gt_ids": gt_ids,
                    "acceptable_ids": acceptable_ids or gt_ids,
                    "hard_neg_ids": hard_neg_ids,
                }
            )

    return queries


# ────────────────────────────── 임베딩 / reranker 로딩 ──────────────────────────────

def detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def load_bge_m3():
    from sentence_transformers import SentenceTransformer

    device = detect_device()
    print(f"BGE-M3 로딩 중... (device={device})")
    return SentenceTransformer("BAAI/bge-m3", device=device)


def build_reranker(config: dict):
    from sentence_transformers import CrossEncoder

    device = detect_device()
    print(f"reranker '{config['name']}' 로딩 중... (device={device})")
    return CrossEncoder(config["hf_id"], device=device, trust_remote_code=config.get("trust_remote_code", False))


# ────────────────────────────── 유사도 / 랭킹 ──────────────────────────────

def cosine_similarity(vec_a, vec_b) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_faqs(query_vec, faq_vectors: dict[str, list[float]]) -> list[str]:
    scored = [(faq_id, cosine_similarity(query_vec, vec)) for faq_id, vec in faq_vectors.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [faq_id for faq_id, _ in scored]


def first_rank(ranked_ids: list[str], target_ids: list[str]) -> int | None:
    for pos, faq_id in enumerate(ranked_ids, start=1):
        if faq_id in target_ids:
            return pos
    return None


def compute_metrics(gt_rank: int | None) -> dict:
    return {
        "rank": gt_rank,
        "top1": gt_rank == 1,
        "top3": gt_rank is not None and gt_rank <= 3,
        "top5": gt_rank is not None and gt_rank <= 5,
        "rr": (1 / gt_rank) if gt_rank else 0.0,
    }


# ────────────────────────────── 평가 파이프라인 ──────────────────────────────

def evaluate_pipeline(reranker_config: dict, faq_corpus: dict[str, str], queries: list[dict], top_k_list: list[int]) -> dict:
    """
    FAQ/쿼리 임베딩은 top_k와 무관하게 딱 한 번만 계산하고,
    top_k별로는 'Top-K 후보를 얼마나 잘라서 reranker에 넘기는지'만 다르게 하여 재사용한다.
    반환값: {top_k: {"summary": [...], "per_query": [...]}}
    """
    name = reranker_config["name"]
    print(f"\n{'=' * 60}\nreranker 평가: {name} (top_k={top_k_list})\n{'=' * 60}")

    embedder = load_bge_m3()

    print(f"FAQ {len(faq_corpus)}개 임베딩 중...")
    faq_ids = list(faq_corpus.keys())
    faq_texts = list(faq_corpus.values())

    faq_embed_start = time.perf_counter()
    faq_vecs = embedder.encode(faq_texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    faq_embed_time = time.perf_counter() - faq_embed_start
    print(f"FAQ {len(faq_corpus)}개 임베딩 완료: {faq_embed_time:.2f}초")

    faq_vectors = dict(zip(faq_ids, faq_vecs))

    reranker = build_reranker(reranker_config)

    # top_k별로 결과를 따로 모은다.
    per_query_by_k = {k: [] for k in top_k_list}
    embed_latencies = []
    rerank_latencies_by_k = {k: [] for k in top_k_list}
    max_k = max(top_k_list)

    print(f"테스트 쿼리 {len(queries)}개 평가 중...")
    for i, q in enumerate(queries, start=1):
        start = time.perf_counter()
        query_vec = embedder.encode(q["query"], normalize_embeddings=True)
        embed_latencies.append(time.perf_counter() - start)

        full_ranked = rank_faqs(query_vec, faq_vectors)
        baseline_rank = first_rank(full_ranked, q["gt_ids"])
        baseline_metrics = compute_metrics(baseline_rank)

        # 가장 큰 top_k만큼만 한 번 잘라두고, 작은 top_k는 이 리스트의 앞부분을 재사용한다.
        max_k_ids = full_ranked[:max_k]

        for top_k in top_k_list:
            top_k_ids = max_k_ids[:top_k]

            start = time.perf_counter()
            pairs = [[q["query"], faq_corpus[fid]] for fid in top_k_ids]
            scores = reranker.predict(pairs)
            rerank_latencies_by_k[top_k].append(time.perf_counter() - start)

            reranked_ids = [fid for fid, _ in sorted(zip(top_k_ids, scores), key=lambda x: x[1], reverse=True)]
            reranked_rank = first_rank(reranked_ids, q["gt_ids"])
            reranked_metrics = compute_metrics(reranked_rank)

            if baseline_rank is None or baseline_rank > top_k:
                change = "1단계 실패(Top-K 밖, reranker 무관)"
            elif reranked_rank is None:
                change = "오류(예상치 못한 결과, 확인 필요)"
            elif reranked_rank < baseline_rank:
                change = "개선"
            elif reranked_rank > baseline_rank:
                change = "악화"
            else:
                change = "동일"

            per_query_by_k[top_k].append(
                {
                    "reranker": name,
                    "test_id": q["test_id"],
                    "category": q["category"],
                    "query": q["query"],
                    "baseline_rank": baseline_rank,
                    "reranked_rank": reranked_rank,
                    "change": change,
                    "baseline_top1": baseline_metrics["top1"],
                    "reranked_top1": reranked_metrics["top1"],
                    "baseline_rr": baseline_metrics["rr"],
                    "reranked_rr": reranked_metrics["rr"],
                }
            )

        if i % 10 == 0:
            print(f"  {i}/{len(queries)} 완료")

    results = {}
    for top_k in top_k_list:
        summary = summarize(
            name, per_query_by_k[top_k], embed_latencies, rerank_latencies_by_k[top_k], top_k, faq_embed_time
        )
        results[top_k] = {"summary": summary, "per_query": per_query_by_k[top_k]}

    return results


def summarize(name: str, results: list[dict], embed_latencies, rerank_latencies, top_k: int, faq_embed_time: float = None) -> list[dict]:
    categories = ["전체"] + sorted({r["category"] for r in results})
    rows = []

    for cat in categories:
        subset = results if cat == "전체" else [r for r in results if r["category"] == cat]
        if not subset:
            continue

        n = len(subset)

        def rate(key):
            return sum(1 for r in subset if r[key]) / n

        row = {
            "reranker": name,
            "category": cat,
            "n": n,
            "top_k": top_k,
            "baseline_top1": round(rate("baseline_top1"), 3),
            "reranked_top1": round(rate("reranked_top1"), 3),
            "baseline_mrr": round(sum(r["baseline_rr"] for r in subset) / n, 3),
            "reranked_mrr": round(sum(r["reranked_rr"] for r in subset) / n, 3),
            "n_improved": sum(1 for r in subset if r["change"] == "개선"),
            "n_worsened": sum(1 for r in subset if r["change"] == "악화"),
            "n_unchanged": sum(1 for r in subset if r["change"] == "동일"),
            "n_stage1_miss": sum(1 for r in subset if r["change"].startswith("1단계 실패")),
        }
        if cat == "전체":
            row["avg_embed_latency_sec"] = round(sum(embed_latencies) / len(embed_latencies), 5)
            row["avg_rerank_latency_sec"] = round(sum(rerank_latencies) / len(rerank_latencies), 5)
            if faq_embed_time is not None:
                row["faq_corpus_embed_time_sec"] = round(faq_embed_time, 3)
        rows.append(row)

    return rows


# ────────────────────────────── 저장 (이어쓰기 + 회차 기록) ──────────────────────────────

def get_next_run_number(summary_path: str) -> int:
    if not os.path.exists(summary_path):
        return 1
    try:
        with open(summary_path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        return max(int(r["run"]) for r in rows) + 1 if rows else 1
    except Exception:
        return 1


def append_summary(summary_path: str, run: int, timestamp: str, result_rows: list[dict]) -> None:
    fieldnames = [
        "run", "timestamp", "reranker", "category", "n", "top_k",
        "baseline_top1", "reranked_top1", "baseline_mrr", "reranked_mrr",
        "n_improved", "n_worsened", "n_unchanged", "n_stage1_miss",
        "avg_embed_latency_sec", "avg_rerank_latency_sec", "faq_corpus_embed_time_sec",
    ]

    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            existing_header = reader.fieldnames or []
            existing_rows = list(reader)
        if existing_header != fieldnames:
            merged = list(existing_header)
            for name in fieldnames:
                if name not in merged:
                    merged.append(name)
            with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=merged)
                writer.writeheader()
                for old_row in existing_rows:
                    writer.writerow({k: old_row.get(k, "") for k in merged})
                for row in result_rows:
                    row_with_meta = {"run": run, "timestamp": timestamp, **row}
                    writer.writerow({k: row_with_meta.get(k, "") for k in merged})
            print(f"[안내] '{summary_path}'의 컬럼 구성이 이전 실행과 달라 컬럼을 통합해 재정렬했습니다.")
            return

    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in result_rows:
            row_with_meta = {"run": run, "timestamp": timestamp, **row}
            writer.writerow({k: row_with_meta.get(k, "") for k in fieldnames})


def append_detail_log(log_path: str, run: int, timestamp: str, reranker_name: str, per_query: list[dict]) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 70}\n")
        f.write(f"회차 {run} | {timestamp} | reranker = {reranker_name}\n")
        f.write(f"{'=' * 70}\n")
        for r in per_query:
            f.write(
                f"[{r['change']}] {r['test_id']} ({r['category']}) "
                f"baseline_rank={r['baseline_rank']} -> reranked_rank={r['reranked_rank']} "
                f"| {r['query']}\n"
            )


# ────────────────────────────── 메인 ──────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="임베딩 단독 vs 임베딩+reranker 검색 정확도를 비교합니다.")
    parser.add_argument("--xlsx", required=True, help="FAQ + 테스트 쿼리가 담긴 xlsx 파일 경로")
    parser.add_argument(
        "--top-k",
        default="3,5,10,15,20",
        help="reranker에 넘길 후보 개수. 쉼표로 여러 값을 지정하면 한 번에 다 비교 (기본: 3,5,10,15,20)",
    )
    parser.add_argument(
        "--rerankers",
        default=None,
        help="쉼표로 구분한 reranker 이름만 선택 실행 (생략 시 RERANKER_CONFIGS 전체 실행)",
    )
    parser.add_argument("--output", default="reranker_comparison", help="결과 저장 파일 접두사")
    args = parser.parse_args()

    top_k_list = [int(x.strip()) for x in args.top_k.split(",")]

    faq_corpus = load_faq_corpus(args.xlsx)
    queries = load_test_queries(args.xlsx)
    print(f"FAQ {len(faq_corpus)}개, FAQ_RAG 테스트 쿼리 {len(queries)}개를 불러왔습니다. (top_k={top_k_list})")

    configs = RERANKER_CONFIGS
    if args.rerankers:
        selected = {x.strip() for x in args.rerankers.split(",")}
        configs = [c for c in RERANKER_CONFIGS if c["name"] in selected]

    summary_path = f"{args.output}_summary.csv"
    detail_path = f"{args.output}_detail.log"
    run = get_next_run_number(summary_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"이번 실행 회차: {run}회차 ({timestamp})")

    all_summaries = []  # (reranker_name, top_k, summary) 튜플 모음
    succeeded, failed = [], []

    for config in configs:
        try:
            results_by_k = evaluate_pipeline(config, faq_corpus, queries, top_k_list)
        except Exception as e:
            print(f"\n[실패] '{config['name']}' 평가 중 오류: {e}")
            failed.append(config["name"])
            continue

        for top_k in top_k_list:
            result = results_by_k[top_k]
            append_summary(summary_path, run, timestamp, result["summary"])
            append_detail_log(detail_path, run, timestamp, f"{config['name']} (top_k={top_k})", result["per_query"])
            all_summaries.append((config["name"], top_k, result["summary"]))

        succeeded.append(config["name"])
        print(f"'{config['name']}' 결과(top_k={top_k_list} 전체)를 {summary_path} / {detail_path}에 저장했습니다.")

    print(f"\n{'=' * 70}")
    print(f"{run}회차 결과 저장 완료. 성공: {succeeded}" + (f" / 실패: {failed}" if failed else ""))
    print(f"{'=' * 70}")

    print(
        f"\n{'reranker':<20}{'top_k':>6}{'Top-1(단독)':>13}{'Top-1(rerank)':>15}"
        f"{'MRR(단독)':>11}{'MRR(rerank)':>13}{'개선/악화/1단계실패':>18}"
    )
    for name, top_k, summary in all_summaries:
        row = next(r for r in summary if r["category"] == "전체")
        change_str = f"{row['n_improved']}/{row['n_worsened']}/{row['n_stage1_miss']}"
        print(
            f"{name:<20}{top_k:>6}{row['baseline_top1']:>13}{row['reranked_top1']:>15}"
            f"{row['baseline_mrr']:>11}{row['reranked_mrr']:>13}{change_str:>18}"
        )


if __name__ == "__main__":
    main()

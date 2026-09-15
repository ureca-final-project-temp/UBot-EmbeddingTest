"""
BGE-M3 임베딩 단독 검색 vs BGE-M3 + reranker(2단계) 검색을 비교하는 스크립트.

방식 (v2 - 후보 폭(buffer)과 최종 개수(k)를 분리):
  (A) 임베딩만 사용: 임베딩 순위 그대로 상위 final_k개를 채택. reranker 없음.
  (B) 임베딩 + reranker: 임베딩으로 final_k + buffer개(버퍼만큼 더 넓게)를 뽑아서
      그 전체를 reranker로 재정렬한 뒤, 재정렬된 순서에서 상위 final_k개만 최종 채택한다.

  즉 buffer=0이면 "reranker가 정확히 final_k개만 보고 그대로 final_k개를 내놓는" 기존 방식과 같고,
  buffer>0이면 "reranker에게 여유 후보를 더 주고, 그중에서 골라내게 한 뒤 다시 final_k로 자르는" 방식이다.
  이렇게 하면 "reranker에게 여유 후보를 더 줬을 때 정말 final_k 안으로 정답을 더 잘 끌어올리는지"를
  직접 검증할 수 있다.

  MAP_API 의도로 표시된 쿼리(예: E09, M09, H09 - 지도 라우팅 테스트)는 FAQ 검색 비교와
  무관하므로 자동으로 제외하고 FAQ_RAG 의도 쿼리만 사용한다.

사전 준비:
  pip install sentence-transformers torch scikit-learn openpyxl

사용법:
  python compare_embedding_reranker.py --xlsx EmbeddingTestFAQ_통합.xlsx
  (기본으로 final_k=5,10 x buffer=0,5,10,15 조합을 전부 테스트합니다.)

  python compare_embedding_reranker.py --xlsx EmbeddingTestFAQ_통합.xlsx --final-k 5 --buffer 0,10,20
  python compare_embedding_reranker.py --xlsx EmbeddingTestFAQ_통합.xlsx --rerankers bge-reranker-v2-m3
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


def hit_at_k(rank: int | None, k: int) -> bool:
    """정답 순위가 k 이내(1..k)에 들었는지 여부."""
    return rank is not None and rank <= k


def reciprocal_rank_at_k(rank: int | None, k: int) -> float:
    """k 이내에 들었을 때만 1/순위 점수를 주고, k를 벗어나면 0점 처리한다.
    (최종적으로 사용자에게 보여지는 게 상위 k개뿐이라는 전제)"""
    if rank is not None and rank <= k:
        return 1 / rank
    return 0.0


# ────────────────────────────── 평가 파이프라인 ──────────────────────────────

def evaluate_pipeline(
    reranker_config: dict,
    faq_corpus: dict[str, str],
    queries: list[dict],
    final_k_list: list[int],
    buffer_list: list[int],
) -> dict:
    """
    FAQ/쿼리 임베딩은 조합과 무관하게 딱 한 번만 계산하고,
    (final_k, buffer) 조합별로는 'reranker에게 몇 개(final_k+buffer)를 보여주고,
    그중 상위 final_k개만 최종 채택하는지'만 다르게 하여 재사용한다.
    반환값: {(final_k, buffer): {"summary": [...], "per_query": [...]}}
    """
    name = reranker_config["name"]
    combos = [(k, b) for k in final_k_list for b in buffer_list]
    print(f"\n{'=' * 60}\nreranker 평가: {name} (final_k x buffer 조합: {combos})\n{'=' * 60}")

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

    per_query_by_combo = {combo: [] for combo in combos}
    embed_latencies = []
    rerank_latencies_by_combo = {combo: [] for combo in combos}
    max_window = max(k + b for k, b in combos)

    print(f"테스트 쿼리 {len(queries)}개 평가 중...")
    for i, q in enumerate(queries, start=1):
        start = time.perf_counter()
        query_vec = embedder.encode(q["query"], normalize_embeddings=True)
        embed_latencies.append(time.perf_counter() - start)

        full_ranked = rank_faqs(query_vec, faq_vectors)
        baseline_rank = first_rank(full_ranked, q["gt_ids"])

        # 가장 큰 후보 폭(final_k+buffer)만큼만 한 번 잘라두고, 작은 조합은 앞부분을 재사용한다.
        max_window_ids = full_ranked[:max_window]

        for final_k, buffer in combos:
            window_size = final_k + buffer
            window_ids = max_window_ids[:window_size]

            start = time.perf_counter()
            pairs = [[q["query"], faq_corpus[fid]] for fid in window_ids]
            scores = reranker.predict(pairs)
            rerank_latencies_by_combo[(final_k, buffer)].append(time.perf_counter() - start)

            reranked_window = [fid for fid, _ in sorted(zip(window_ids, scores), key=lambda x: x[1], reverse=True)]
            # 재정렬된 후보 중 상위 final_k개만 최종 채택 (그 밖은 버림)
            reranked_rank_in_window = first_rank(reranked_window, q["gt_ids"])

            baseline_hit = hit_at_k(baseline_rank, final_k)
            reranked_hit = hit_at_k(reranked_rank_in_window, final_k)

            if baseline_rank is None or baseline_rank > window_size:
                # 임베딩이 애초에 후보 폭(final_k+buffer) 안에도 못 넣은 경우. reranker가 손쓸 여지 없음.
                change = "1단계 실패(후보 폭 밖, reranker 무관)"
            elif reranked_rank_in_window is None:
                change = "오류(예상치 못한 결과, 확인 필요)"
            elif baseline_hit and not reranked_hit:
                change = "악화(k 안에 있었는데 밀려남)"
            elif not baseline_hit and reranked_hit:
                change = "개선(k 밖이었는데 들어옴)"
            elif baseline_hit and reranked_hit and reranked_rank_in_window < baseline_rank:
                change = "개선(순위 상승)"
            elif baseline_hit and reranked_hit and reranked_rank_in_window > baseline_rank:
                change = "악화(순위 하락, k 안에서는 유지)"
            else:
                change = "동일"

            per_query_by_combo[(final_k, buffer)].append(
                {
                    "reranker": name,
                    "test_id": q["test_id"],
                    "category": q["category"],
                    "query": q["query"],
                    "baseline_rank": baseline_rank,
                    "reranked_rank": reranked_rank_in_window,
                    "change": change,
                    "baseline_hit": baseline_hit,
                    "reranked_hit": reranked_hit,
                    "baseline_rr": reciprocal_rank_at_k(baseline_rank, final_k),
                    "reranked_rr": reciprocal_rank_at_k(reranked_rank_in_window, final_k),
                }
            )

        if i % 10 == 0:
            print(f"  {i}/{len(queries)} 완료")

    results = {}
    for final_k, buffer in combos:
        summary = summarize(
            name,
            per_query_by_combo[(final_k, buffer)],
            embed_latencies,
            rerank_latencies_by_combo[(final_k, buffer)],
            final_k,
            buffer,
            faq_embed_time,
        )
        results[(final_k, buffer)] = {"summary": summary, "per_query": per_query_by_combo[(final_k, buffer)]}

    return results


def summarize(
    name: str,
    results: list[dict],
    embed_latencies,
    rerank_latencies,
    final_k: int,
    buffer: int,
    faq_embed_time: float = None,
) -> list[dict]:
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
            "final_k": final_k,
            "buffer": buffer,
            "rerank_window": final_k + buffer,
            "baseline_hit_rate": round(rate("baseline_hit"), 3),
            "reranked_hit_rate": round(rate("reranked_hit"), 3),
            "baseline_mrr": round(sum(r["baseline_rr"] for r in subset) / n, 3),
            "reranked_mrr": round(sum(r["reranked_rr"] for r in subset) / n, 3),
            "n_improved": sum(1 for r in subset if r["change"].startswith("개선")),
            "n_worsened": sum(1 for r in subset if r["change"].startswith("악화")),
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
        "run", "timestamp", "reranker", "category", "n", "final_k", "buffer", "rerank_window",
        "baseline_hit_rate", "reranked_hit_rate", "baseline_mrr", "reranked_mrr",
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
        "--final-k",
        default="5,10",
        help="최종적으로 채택할 결과 개수(k). 쉼표로 여러 값 지정 가능 (기본: 5,10)",
    )
    parser.add_argument(
        "--buffer",
        default="0,5,10,15",
        help="reranker에게 final_k보다 얼마나 더 넓게(n) 후보를 보여줄지. "
        "buffer=0이면 기존 방식(정확히 final_k개만 봄)과 동일 (기본: 0,5,10,15)",
    )
    parser.add_argument(
        "--rerankers",
        default=None,
        help="쉼표로 구분한 reranker 이름만 선택 실행 (생략 시 RERANKER_CONFIGS 전체 실행)",
    )
    parser.add_argument("--output", default="reranker_comparison", help="결과 저장 파일 접두사")
    args = parser.parse_args()

    final_k_list = [int(x.strip()) for x in args.final_k.split(",")]
    buffer_list = [int(x.strip()) for x in args.buffer.split(",")]

    faq_corpus = load_faq_corpus(args.xlsx)
    queries = load_test_queries(args.xlsx)
    print(
        f"FAQ {len(faq_corpus)}개, FAQ_RAG 테스트 쿼리 {len(queries)}개를 불러왔습니다. "
        f"(final_k={final_k_list}, buffer={buffer_list})"
    )

    configs = RERANKER_CONFIGS
    if args.rerankers:
        selected = {x.strip() for x in args.rerankers.split(",")}
        configs = [c for c in RERANKER_CONFIGS if c["name"] in selected]

    summary_path = f"{args.output}_summary.csv"
    detail_path = f"{args.output}_detail.log"
    run = get_next_run_number(summary_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"이번 실행 회차: {run}회차 ({timestamp})")

    all_summaries = []  # (reranker_name, final_k, buffer, summary) 튜플 모음
    succeeded, failed = [], []

    for config in configs:
        try:
            results_by_combo = evaluate_pipeline(config, faq_corpus, queries, final_k_list, buffer_list)
        except Exception as e:
            print(f"\n[실패] '{config['name']}' 평가 중 오류: {e}")
            failed.append(config["name"])
            continue

        for (final_k, buffer), result in results_by_combo.items():
            append_summary(summary_path, run, timestamp, result["summary"])
            append_detail_log(
                detail_path, run, timestamp, f"{config['name']} (final_k={final_k}, buffer={buffer})", result["per_query"]
            )
            all_summaries.append((config["name"], final_k, buffer, result["summary"]))

        succeeded.append(config["name"])
        print(f"'{config['name']}' 결과(전체 조합)를 {summary_path} / {detail_path}에 저장했습니다.")

    print(f"\n{'=' * 70}")
    print(f"{run}회차 결과 저장 완료. 성공: {succeeded}" + (f" / 실패: {failed}" if failed else ""))
    print(f"{'=' * 70}")

    print(
        f"\n{'reranker':<20}{'final_k':>8}{'buffer':>7}{'Hit(단독)':>10}{'Hit(rerank)':>12}"
        f"{'MRR(단독)':>10}{'MRR(rerank)':>12}{'개선/악화':>12}"
    )
    for name, final_k, buffer, summary in all_summaries:
        row = next(r for r in summary if r["category"] == "전체")
        change_str = f"{row['n_improved']}/{row['n_worsened']}"
        print(
            f"{name:<20}{final_k:>8}{buffer:>7}{row['baseline_hit_rate']:>10}{row['reranked_hit_rate']:>12}"
            f"{row['baseline_mrr']:>10}{row['reranked_mrr']:>12}{change_str:>12}"
        )


if __name__ == "__main__":
    main()

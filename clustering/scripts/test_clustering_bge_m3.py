"""
BGE-M3 임베딩 + HDBSCAN 클러스터링이 실제로 잘 동작하는지 검증하는 스크립트.

방식:
  EmbeddingTestFAQ_통합.xlsx의 '미등록 클러스터링' 시트를 읽어서, 이미 정답 주제(기대 클러스터)가
  라벨링된 질문들을 BGE-M3로 임베딩한 뒤 HDBSCAN(sklearn)으로 클러스터링하고, 실제 클러스터 결과가
  정답 주제와 얼마나 일치하는지 여러 지표로 정량 평가합니다.

  - ARI/NMI: 1에 가까울수록 "정답 그룹과 실제 클러스터가 거의 똑같다"는 뜻
  - Homogeneity: 1에 가까울수록 클러스터 안에 서로 다른 정답 주제가 섞이지 않았다는 뜻
                 (여러 주제가 한 클러스터로 뭉치는 문제를 잡아냄)
  - Completeness: 1에 가까울수록 같은 정답 주제가 여러 클러스터로 쪼개지지 않았다는 뜻
                  (같은 주제가 과도하게 세분화되는 문제를 잡아냄)
  - 평균 확신도(probabilities_): 각 점이 자기 클러스터에 얼마나 확실하게 속하는지 (0~1)
  - "노이즈(무관)"으로 라벨링된 질문이 실제로 어느 클러스터에도 안 묶이고 노이즈로 남는지 확인합니다.
  - "A / B"처럼 애매한 경계 케이스로 라벨링된 질문은 평가상 첫 번째 주제(A)를 정답으로 간주하되,
    출력에는 원래 라벨을 그대로 보여줘서 애매한 케이스였음을 알 수 있게 합니다.

사전 준비:
  pip install sentence-transformers torch scikit-learn openpyxl

사용법:
  python test_clustering_bge_m3.py
  python test_clustering_bge_m3.py --min-cluster-size 2,3,4
  python test_clustering_bge_m3.py --xlsx EmbeddingTestFAQ_통합.xlsx --sheet 미등록 클러스터링
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime


def load_cluster_test_set(xlsx_path: str, sheet_name: str):
    """
    '미등록 클러스터링' 시트를 읽어서 (질문 리스트, 정답 topic id 리스트, topic id->이름 dict)를 반환한다.

    시트 컬럼: ID, 사용자 질문, 기대 클러스터, 애매성, 처리 의도, 세부 의도, 설계 목적
    '기대 클러스터'가 '노이즈(무관)'이면 topic id = -1.
    'A / B'처럼 두 클러스터에 걸친 애매한 라벨은 첫 번째(A)를 평가용 정답으로 사용한다.
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    if sheet_name not in wb.sheetnames:
        print(f"[오류] '{sheet_name}' 시트를 찾을 수 없습니다. 시트 목록: {wb.sheetnames}")
        sys.exit(1)
    ws = wb[sheet_name]

    questions, display_labels = [], []
    topic_name_to_id = {}
    true_topics = []
    next_id = 0

    for row in ws.iter_rows(min_row=2, values_only=True):
        question, expected_cluster = row[1], row[2]
        if not question:
            continue

        display_labels.append(expected_cluster)

        if expected_cluster and expected_cluster.startswith("노이즈"):
            true_topics.append(-1)
        else:
            # "발신자 정보 표시 / 스팸 차단" 같은 애매한 라벨은 첫 번째 주제를 정답으로 사용
            primary_topic = (expected_cluster or "").split("/")[0].strip()
            if primary_topic not in topic_name_to_id:
                topic_name_to_id[primary_topic] = next_id
                next_id += 1
            true_topics.append(topic_name_to_id[primary_topic])

        questions.append(question)

    topic_id_to_name = {v: k for k, v in topic_name_to_id.items()}
    topic_id_to_name[-1] = "(노이즈)"

    return questions, true_topics, topic_id_to_name, display_labels


def load_bge_m3():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[오류] sentence-transformers가 설치되어 있지 않습니다. pip install sentence-transformers torch")
        sys.exit(1)

    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    print(f"BGE-M3 로딩 중... (device={device})")
    return SentenceTransformer("BAAI/bge-m3", device=device)


def run_clustering(embeddings, min_cluster_size: int):
    from sklearn.cluster import HDBSCAN
    import numpy as np

    # 코사인 거리를 쓰기 위해 벡터를 미리 정규화(길이를 1로) 하면
    # 유클리드 거리가 코사인 거리와 사실상 동일해진다.
    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # BGE-M3 클러스터링 품질 자체를 검증하는 게 목적이므로, 검증된 sklearn 구현을 사용한다.
    # (원조 hdbscan 패키지는 outlier_scores_/cluster_persistence_도 제공하지만,
    #  내부 알고리즘 차이로 결과가 달라질 수 있어 제외하고 확신도만 사용한다.)
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", copy=True)

    start = time.perf_counter()
    labels = clusterer.fit_predict(norm_embeddings)
    elapsed = time.perf_counter() - start

    # probabilities_: 각 점이 자기 클러스터에 얼마나 확실하게 속하는지 (0~1, 노이즈는 0)
    extras = {"probabilities": clusterer.probabilities_}

    return labels, elapsed, extras


def evaluate(true_topics, predicted_labels):
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        homogeneity_score,
        completeness_score,
    )

    ari = adjusted_rand_score(true_topics, predicted_labels)
    nmi = normalized_mutual_info_score(true_topics, predicted_labels)
    # homogeneity: 각 클러스터가 단일 정답 주제로만 구성됐는지 (서로 다른 주제가 섞이면 낮아짐)
    homogeneity = homogeneity_score(true_topics, predicted_labels)
    # completeness: 같은 정답 주제가 여러 클러스터로 쪼개지지 않았는지 (과도한 세분화면 낮아짐)
    completeness = completeness_score(true_topics, predicted_labels)
    return ari, nmi, homogeneity, completeness


def print_clusters(questions, true_topics, predicted_labels, topic_names, display_labels, extras):
    probabilities = extras["probabilities"]

    lines = ["\n[클러스터별 실제 결과]"]
    groups = defaultdict(list)
    for q, t, pred, disp, prob in zip(questions, true_topics, predicted_labels, display_labels, probabilities):
        groups[pred].append((q, t, disp, prob))

    for cluster_id in sorted(groups.keys(), key=lambda x: (x == -1, x)):
        members = groups[cluster_id]

        if cluster_id == -1:
            lines.append(f"\n  노이즈로 분류됨 ({len(members)}개)")
        else:
            avg_prob = sum(m[3] for m in members) / len(members)
            lines.append(f"\n  클러스터 {cluster_id} ({len(members)}개) | 평균 확신도: {avg_prob:.2f}")

        for q, true_topic, disp, prob in members:
            mark = "✓" if true_topic != -1 else "·"
            if cluster_id == -1:
                lines.append(f"    {mark} [{disp}] {q}")
            else:
                lines.append(f"    {mark} [{disp}] {q}  (확신도: {prob:.2f})")

    text = "\n".join(lines)
    print(text)
    return text


def get_next_run_number(summary_path: str) -> int:
    """요약 CSV의 마지막 회차 번호를 읽어서 +1한 값을 반환한다. 파일이 없으면 1부터 시작."""
    if not os.path.exists(summary_path):
        return 1
    try:
        with open(summary_path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return 1
        return max(int(r["run"]) for r in rows) + 1
    except Exception:
        return 1


def append_summary(summary_path: str, run: int, timestamp: str, result_rows: list[dict]) -> None:
    fieldnames = [
        "run", "timestamp", "min_cluster_size", "n_clusters", "n_true_topics",
        "n_noise", "n_true_noise", "ari", "nmi", "homogeneity", "completeness",
        "avg_probability", "clustering_time_sec",
    ]
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in result_rows:
            row_with_meta = {"run": run, "timestamp": timestamp, **row}
            writer.writerow({k: row_with_meta.get(k, "") for k in fieldnames})


def append_detail_log(log_path: str, run: int, timestamp: str, min_cluster_size: int, log_text: str) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 70}\n")
        f.write(f"회차 {run} | {timestamp} | min_cluster_size = {min_cluster_size}\n")
        f.write(f"{'=' * 70}\n")
        f.write(log_text)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="BGE-M3 임베딩 + HDBSCAN 클러스터링 품질을 검증합니다.")
    parser.add_argument(
        "--xlsx",
        default="EmbeddingTestFAQ_통합.xlsx",
        help="테스트 데이터가 담긴 xlsx 파일 경로 (기본: EmbeddingTestFAQ_통합.xlsx)",
    )
    parser.add_argument(
        "--sheet",
        default="미등록 클러스터링",
        help="클러스터링 테스트 데이터가 있는 시트 이름 (기본: 미등록 클러스터링)",
    )
    parser.add_argument(
        "--min-cluster-size",
        default="2,3,4,5,6",
        help="HDBSCAN의 min_cluster_size 값. 쉼표로 여러 개 지정하면 각각 비교 (기본: 2,3,4,5,6)",
    )
    parser.add_argument(
        "--output",
        default="clustering_results",
        help="결과 저장 파일 접두사. {output}_summary.csv / {output}_detail.log로 저장 (기본: clustering_results)",
    )
    args = parser.parse_args()

    min_sizes = [int(x.strip()) for x in args.min_cluster_size.split(",")]
    summary_path = f"{args.output}_summary.csv"
    detail_path = f"{args.output}_detail.log"

    run = get_next_run_number(summary_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"이번 실행 회차: {run}회차 ({timestamp})")

    questions, true_topics, topic_names, display_labels = load_cluster_test_set(args.xlsx, args.sheet)
    n_true_topics = len({t for t in true_topics if t != -1})
    n_true_noise = true_topics.count(-1)
    print(f"'{args.sheet}' 시트에서 질문 {len(questions)}개 로드 완료")
    print(f"정답 주제 수: {n_true_topics}개, 정답 노이즈 수: {n_true_noise}개")

    model = load_bge_m3()
    print(f"질문 {len(questions)}개 임베딩 중...")
    embeddings = model.encode(questions, normalize_embeddings=True)

    result_rows = []

    for min_size in min_sizes:
        print(f"\n{'=' * 60}")
        print(f"min_cluster_size = {min_size}")
        print(f"{'=' * 60}")

        predicted_labels, clustering_time, extras = run_clustering(embeddings, min_size)
        ari, nmi, homogeneity, completeness = evaluate(true_topics, predicted_labels)

        n_clusters = len(set(predicted_labels)) - (1 if -1 in predicted_labels else 0)
        n_noise = list(predicted_labels).count(-1)

        # 노이즈를 제외한 점들의 평균 확신도
        non_noise_probs = [p for p, l in zip(extras["probabilities"], predicted_labels) if l != -1]
        avg_probability = sum(non_noise_probs) / len(non_noise_probs) if non_noise_probs else 0.0

        print(f"생성된 클러스터 수: {n_clusters} (정답 주제 수: {n_true_topics})")
        print(f"노이즈로 분류된 개수: {n_noise} (정답 노이즈 개수: {n_true_noise})")
        print(f"ARI (Adjusted Rand Index): {ari:.3f}  (1에 가까울수록 정답과 일치)")
        print(f"NMI (Normalized Mutual Info): {nmi:.3f}  (1에 가까울수록 정답과 일치)")
        print(f"Homogeneity: {homogeneity:.3f}  (1에 가까울수록 클러스터 내 주제 혼합이 없음)")
        print(f"Completeness: {completeness:.3f}  (1에 가까울수록 같은 주제가 안 쪼개짐)")
        print(f"평균 확신도(노이즈 제외): {avg_probability:.3f}")
        print(f"클러스터링 소요 시간: {clustering_time * 1000:.2f}ms")

        detail_text = print_clusters(questions, true_topics, predicted_labels, topic_names, display_labels, extras)
        append_detail_log(detail_path, run, timestamp, min_size, detail_text)

        result_rows.append(
            {
                "min_cluster_size": min_size,
                "n_clusters": n_clusters,
                "n_true_topics": n_true_topics,
                "n_noise": n_noise,
                "n_true_noise": n_true_noise,
                "ari": round(ari, 3),
                "nmi": round(nmi, 3),
                "homogeneity": round(homogeneity, 3),
                "completeness": round(completeness, 3),
                "avg_probability": round(avg_probability, 3),
                "clustering_time_sec": round(clustering_time, 5),
            }
        )

    append_summary(summary_path, run, timestamp, result_rows)

    print(f"\n{'=' * 60}")
    print(f"{run}회차 결과를 저장했습니다: {summary_path} (요약) / {detail_path} (상세)")
    print(f"{'=' * 60}")

    print("\n[이번 회차 min_cluster_size별 비교]")
    print(f"{'size':>6}{'클러스터':>10}{'노이즈':>8}{'ARI':>8}{'NMI':>8}{'Homog':>8}{'Compl':>8}{'시간(ms)':>10}")
    for row in result_rows:
        print(
            f"{row['min_cluster_size']:>6}{row['n_clusters']:>10}{row['n_noise']:>8}"
            f"{row['ari']:>8}{row['nmi']:>8}{row['homogeneity']:>8}{row['completeness']:>8}"
            f"{row['clustering_time_sec'] * 1000:>10.2f}"
        )


if __name__ == "__main__":
    main()

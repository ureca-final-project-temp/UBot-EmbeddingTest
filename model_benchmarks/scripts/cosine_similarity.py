"""
Ollama 임베딩 모델을 이용해 두 문장의 코사인 유사도를 계산하는 스크립트.

사전 준비:
1. Ollama가 로컬에서 실행 중이어야 합니다 (기본: http://localhost:11434)
2. 임베딩 모델이 미리 받아져 있어야 합니다.
   예) ollama pull qwen3-embedding:0.6b
   (노트북 사양이 좋다면 :4b 또는 :8b도 시도해볼 수 있습니다)
3. requests 라이브러리가 필요합니다.
   pip install requests

사용법 (터미널에서 직접 실행):
    python cosine_similarity.py
    python cosine_similarity.py --model qwen3-embedding:4b
    python cosine_similarity.py --text1 "핸드폰 요금 확인 방법" --text2 "요금 조회는 어떻게 하나요"
"""

import argparse
import math
import sys
import time

import requests

OLLAMA_URL = "http://localhost:11434/api/embed"


def get_embedding(text: str, model: str) -> dict:
    """
    Ollama /api/embed 엔드포인트를 호출해서 임베딩 벡터와 소요 시간을 받아온다.

    반환값:
        {
            "vector": [...],                # 임베딩 벡터
            "client_elapsed": float,        # 클라이언트 기준 왕복 시간(초). 네트워크 지연 포함
            "server_total_duration": float, # 서버가 보고한 총 처리 시간(초)
            "server_load_duration": float,  # 서버가 보고한 모델 로딩 시간(초). 콜드스타트 여부 확인용
        }
    """
    payload = {"model": model, "input": text}

    start = time.perf_counter()
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(
            "[오류] Ollama 서버에 연결할 수 없습니다. "
            "'ollama serve'가 실행 중인지 확인하세요."
        )
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"[오류] Ollama API 호출 실패: {e}")
        print(f"응답 내용: {response.text}")
        sys.exit(1)
    client_elapsed = time.perf_counter() - start

    data = response.json()

    # /api/embed는 embeddings 필드에 [벡터] 형태(리스트의 리스트)로 반환한다.
    embeddings = data.get("embeddings")
    if not embeddings:
        print(f"[오류] 응답에 embeddings 필드가 없습니다: {data}")
        sys.exit(1)

    # Ollama가 함께 반환하는 시간 정보(나노초 단위) -> 초 단위로 변환
    server_total_ns = data.get("total_duration", 0)
    server_load_ns = data.get("load_duration", 0)

    return {
        "vector": embeddings[0],
        "client_elapsed": client_elapsed,
        "server_total_duration": server_total_ns / 1e9,
        "server_load_duration": server_load_ns / 1e9,
    }


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """두 벡터의 코사인 유사도를 계산한다 (외부 라이브러리 없이 순수 파이썬)."""
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"벡터 길이가 다릅니다: {len(vec_a)} vs {len(vec_b)}. "
            "두 임베딩이 같은 모델로 생성되었는지 확인하세요."
        )

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        raise ValueError("벡터의 크기(norm)가 0입니다. 빈 텍스트를 입력하지 않았는지 확인하세요.")

    return dot_product / (norm_a * norm_b)


def main():
    parser = argparse.ArgumentParser(description="두 문장의 임베딩 코사인 유사도를 계산합니다.")
    parser.add_argument("--model", default="qwen3-embedding:0.6b", help="사용할 Ollama 임베딩 모델명")
    parser.add_argument("--text1", default=None, help="첫 번째 문장 (생략 시 직접 입력받음)")
    parser.add_argument("--text2", default=None, help="두 번째 문장 (생략 시 직접 입력받음)")
    args = parser.parse_args()

    text1 = args.text1 or input("첫 번째 문장을 입력하세요: ").strip()
    text2 = args.text2 or input("두 번째 문장을 입력하세요: ").strip()

    print(f"\n[모델: {args.model}]")
    print(f"문장 1: {text1}")
    print(f"문장 2: {text2}\n")

    overall_start = time.perf_counter()

    print("임베딩 생성 중... (문장 1)")
    result1 = get_embedding(text1, args.model)
    print(
        f"  → 클라이언트 왕복: {result1['client_elapsed']:.3f}초 "
        f"(서버 처리: {result1['server_total_duration']:.3f}초, "
        f"모델 로딩: {result1['server_load_duration']:.3f}초)"
    )

    print("임베딩 생성 중... (문장 2)")
    result2 = get_embedding(text2, args.model)
    print(
        f"  → 클라이언트 왕복: {result2['client_elapsed']:.3f}초 "
        f"(서버 처리: {result2['server_total_duration']:.3f}초, "
        f"모델 로딩: {result2['server_load_duration']:.3f}초)"
    )

    vec1 = result1["vector"]
    vec2 = result2["vector"]

    print(f"\n벡터 차원: {len(vec1)}")

    similarity_start = time.perf_counter()
    similarity = cosine_similarity(vec1, vec2)
    similarity_elapsed = time.perf_counter() - similarity_start

    overall_elapsed = time.perf_counter() - overall_start

    print(f"\n코사인 유사도: {similarity:.4f}")
    print(f"유사도 계산 시간: {similarity_elapsed * 1000:.2f}ms (거의 무시할 수준)")
    print(f"전체 소요 시간(임베딩 2회 + 계산 포함): {overall_elapsed:.3f}초")

    if similarity > 0.8:
        print("→ 매우 유사한 문장입니다.")
    elif similarity > 0.5:
        print("→ 어느 정도 관련 있는 문장입니다.")
    else:
        print("→ 관련성이 낮은 문장입니다.")


if __name__ == "__main__":
    main()

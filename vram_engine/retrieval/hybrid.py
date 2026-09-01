"""BM25 + Spectral 결합기.

이 클래스는 오직 "두 점수를 정규화해서 가중합한다"만 한다. 명시적으로
하지 않는 것:
  * verifier 호출 안 함
  * circuit breaker 호출 안 함
  * graph state 변경(mutate) 안 함
  * 최종 답변 생성 안 함
  * Agent 판단 안 함
이 책임들은 전부 agent/agent.py 쪽으로 이동했다.

노이즈 플로어: 스펙트럴 유사도의 실제 값이 1e-7~1e-8 수준(사실상 수치
노이즈)인 경우가 있는데, 그냥 min-max 정규화를 하면 0~1 전체 범위로
증폭되어 BM25 같은 진짜 신호보다 랭킹에 더 큰 영향을 주게 된다. 편차가
noise_floor보다 작으면 중립값(0.5)으로 처리해 이 문제를 막는다.
(원본 hybrid_retriever._minmax와 동일 로직, 값 1e-3 유지)
"""
from typing import List

from vram_engine.core.document_store import DocumentStore
from vram_engine.core.graph_store import GraphStore
from vram_engine.core.types import RetrievalCandidate
from vram_engine.retrieval.bm25 import BM25Retriever
from vram_engine.retrieval.spectral import SpectralRetriever

BM25_NOISE_FLOOR = 1e-9
SPECTRAL_NOISE_FLOOR = 1e-3


def minmax_normalize(values: List[float], noise_floor: float = 1e-9) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < noise_floor:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, spectral: SpectralRetriever, alpha: float = 0.5):
        self.bm25 = bm25
        self.spectral = spectral
        self.alpha = alpha  # BM25:스펙트럴 가중치 (0=스펙트럴만, 1=BM25만)

    def retrieve(
        self, query_text: str, document_store: DocumentStore, graph_store: GraphStore, top_k: int
    ) -> List[RetrievalCandidate]:
        documents = document_store.all()

        bm25_scores = self.bm25.score(query_text)
        spectral_scores = self.spectral.score(query_text, documents, graph_store)

        bm25_norm = minmax_normalize(bm25_scores, BM25_NOISE_FLOOR)
        spectral_norm = minmax_normalize(spectral_scores, SPECTRAL_NOISE_FLOOR)

        candidates = []
        for doc in documents:
            i = doc.doc_id
            hybrid_score = self.alpha * bm25_norm[i] + (1 - self.alpha) * spectral_norm[i]
            candidates.append(RetrievalCandidate(
                doc_id=i,
                bm25_score=bm25_scores[i],
                spectral_score=spectral_scores[i],
                hybrid_score=hybrid_score,
            ))

        candidates.sort(key=lambda c: c.hybrid_score, reverse=True)
        return candidates[:top_k]

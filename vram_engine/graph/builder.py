"""문서 그래프 구축.

DocumentStore의 토큰을 가지고 인접행렬을 만드는 순수 함수만 둔다.
GraphStore는 이 함수들이 만든 텐서를 보관할 뿐, 만드는 방법은 모른다.
(기존 hybrid_retriever._build_doc_graph / _augmented_graph의 그래프 생성
부분만 분리 이식 — 계산식은 변경 없음)
"""
from typing import List

import torch

from vram_engine.core.types import Document


def build_similarity_graph(documents: List[Document], edge_threshold: float = 0.02) -> torch.Tensor:
    """문서 토큰의 자카드 유사도로 문서-문서 그래프를 만든다."""
    n = len(documents)
    adj = torch.eye(n)
    sets = [set(doc.tokens) for doc in documents]
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j]) or 1
            sim = inter / union
            if sim >= edge_threshold:
                adj[i, j] = sim
                adj[j, i] = sim
    return adj


def build_query_edge_weights(query_tokens: List[str], documents: List[Document]) -> List[float]:
    """질의 토큰과 각 문서 간의 자카드 유사도를 엣지 가중치로 계산한다."""
    q_set = set(query_tokens)
    weights = []
    for doc in documents:
        d_set = set(doc.tokens)
        inter = len(q_set & d_set)
        union = len(q_set | d_set) or 1
        weights.append(inter / union)
    return weights

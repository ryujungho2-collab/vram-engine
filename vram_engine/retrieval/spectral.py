"""그래프 구조 기반(스펙트럴) 검색기.

isomorphism.py의 공개 API(compute_laplacian_spectrum)만 사용한다 —
private 메서드에 접근하지 않는다. 질의를 그래프에 새 노드로 얹어 라플라시안
고유공간에 투영하고, 문서 노드들과의 코사인 유사도(절댓값)를 낸다.
(계산식 자체는 원본 hybrid_retriever.query()에서 그대로 이식 — 변경 없음)
"""
from typing import List

import torch
import torch.nn.functional as F

from vram_engine.core.document_store import hybrid_tokenize
from vram_engine.core.graph_store import GraphStore
from vram_engine.core.types import Document
from vram_engine.graph.builder import build_query_edge_weights
from vram_engine.graph.isomorphism import CrossDomainIsomorphismEngine, device as ISO_DEVICE


class SpectralRetriever:
    def __init__(self, iso_engine: CrossDomainIsomorphismEngine, top_k_dims: int = 8):
        if top_k_dims <= 0:
            raise ValueError("top_k_dims must be greater than zero")
        self.iso_engine = iso_engine
        self.top_k_dims = top_k_dims

    def score(self, query_text: str, documents: List[Document], graph_store: GraphStore) -> List[float]:
        """질의를 새 노드로 추가한 그래프에서 각 문서와의 (부호 제거된) 코사인
        유사도를 원점수(정규화 이전)로 반환한다.

        빈 코퍼스에서는 질의 노드만 포함된 1x1 그래프의 고유벡터를 문서
        점수로 해석할 수 없으므로 빈 결과를 반환한다.
        """
        if not documents:
            return []

        query_tokens = hybrid_tokenize(query_text)
        edge_weights = build_query_edge_weights(query_tokens, documents)
        aug = graph_store.augmented_with(edge_weights).to(ISO_DEVICE)

        n = len(documents)
        _e_vals, e_vecs = self.iso_engine.compute_laplacian_spectrum(aug)
        k = min(e_vecs.size(1), self.top_k_dims)
        feats = F.normalize(e_vecs[:, :k], p=2, dim=1)
        q_feat = feats[n]
        doc_feats = feats[:n]

        # 고유벡터 부호 임의성 제거 (isomorphism.py와 동일한 처리)
        return torch.abs(doc_feats @ q_feat).tolist()

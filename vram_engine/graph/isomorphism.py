"""도메인 간 그래프 동형(isomorphism) 매칭 엔진.

알고리즘은 원본과 완전히 동일하다: Laplacian(D-A) → eigh → 상위 k개
고유벡터 L2 정규화 → 내적(cost matrix)에 절댓값(부호 모호성 제거) →
Sinkhorn 반복 정규화로 소프트 매칭.

바뀐 것은 단 하나: `_compute_laplacian_spectrum`(private)을
`compute_laplacian_spectrum`(public)으로 승격했다. retrieval/spectral.py가
이 공개 API를 통해서만 접근한다 (예전처럼 언더스코어 메서드에 직접
접근하지 않음).
"""
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CrossDomainIsomorphismEngine:
    def __init__(self, sinkhorn_iters: int = 20, tau: float = 0.05):
        if tau <= 0:
            raise ValueError("tau must be greater than zero")
        if sinkhorn_iters < 0:
            raise ValueError("sinkhorn_iters must be non-negative")
        self.sinkhorn_iters = sinkhorn_iters
        self.tau = tau

    def compute_laplacian_spectrum(self, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_adjacency(adj, "adj")
        deg = torch.diag(adj.sum(dim=1))
        laplacian = deg - adj
        e_vals, e_vecs = torch.linalg.eigh(laplacian)
        return e_vals, e_vecs

    def match_domains(self, adj_a: torch.Tensor, labels_a: List[str], adj_b: torch.Tensor, labels_b: List[str]) -> List[Dict[str, Any]]:
        self._validate_adjacency(adj_a, "adj_a")
        self._validate_adjacency(adj_b, "adj_b")
        if len(labels_a) != adj_a.size(0) or len(labels_b) != adj_b.size(0):
            raise ValueError("each labels list must match its adjacency size")
        _, e_vecs_a = self.compute_laplacian_spectrum(adj_a.to(device))
        _, e_vecs_b = self.compute_laplacian_spectrum(adj_b.to(device))

        min_k = min(e_vecs_a.size(1), e_vecs_b.size(1))
        feat_a = F.normalize(e_vecs_a[:, :min_k], p=2, dim=1)
        feat_b = F.normalize(e_vecs_b[:, :min_k], p=2, dim=1)

        # 라플라시안 고유벡터는 부호(+/-)가 임의로 결정되므로(sign ambiguity),
        # 내적의 부호도 임의성을 가진다. 유사도의 "크기"만 신뢰하도록 절댓값을 취한다.
        cost_matrix = torch.abs(torch.matmul(feat_a, feat_b.T))
        P = torch.exp(cost_matrix / self.tau)
        for _ in range(self.sinkhorn_iters):
            P = P / (P.sum(dim=1, keepdim=True) + 1e-9)
            P = P / (P.sum(dim=0, keepdim=True) + 1e-9)

        analogies = []
        for i in range(len(labels_a)):
            best_j = int(torch.argmax(P[i, :]).item())
            confidence = float(P[i, best_j].item())
            if confidence > 0.3:
                analogies.append({
                    "domain_A_node": labels_a[i],
                    "domain_B_node": labels_b[best_j],
                    "isomorphism_confidence": round(confidence, 4)
                })
        return analogies

    @staticmethod
    def _validate_adjacency(adj: torch.Tensor, name: str) -> None:
        if not isinstance(adj, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if adj.ndim != 2 or adj.size(0) != adj.size(1):
            raise ValueError(f"{name} must be a square matrix")
        if not torch.isfinite(adj).all().item():
            raise ValueError(f"{name} must not contain NaN or Inf")

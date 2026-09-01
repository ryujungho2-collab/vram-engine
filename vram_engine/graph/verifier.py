"""그래프 레벨 검증기.

엔트로피 공식, ΔH 계산, KL(p_pert‖p_base) 방향 모두 원본과 동일 (알고리즘
변경 없음). 바뀐 것:
1. `_calc_entropy`(private) → `calc_entropy`(public) 승격.
2. `verify_candidates()` 신규 추가 — 예전에는 hybrid_retriever.query() 안에
   for-loop로 흩어져 있던 "후보 문서 전체에 대해 verify_hypothesis 반복 호출"
   로직을 Verifier 내부의 공개 API로 캡슐화했다. Retriever/Agent는 이제
   이 반복문을 직접 구현하지 않는다.
"""
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple

from vram_engine.core.types import VerificationResult

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AntiHallucinationVerifier:
    def __init__(self, entropy_tolerance: float = 0.05, divergence_limit: float = 0.3):
        self.entropy_tolerance = entropy_tolerance
        self.divergence_limit = divergence_limit

    def calc_entropy(self, adj: torch.Tensor) -> torch.Tensor:
        self._validate_adjacency(adj)
        p = F.softmax(adj, dim=-1)
        return -torch.sum(p * torch.log(p + 1e-9)) / adj.size(0)

    def verify_hypothesis(self, base_adj: torch.Tensor, node_u: int, node_v: int) -> Tuple[bool, float, str]:
        self._validate_adjacency(base_adj)
        self._validate_node_index(node_u, base_adj.size(0), "node_u")
        self._validate_node_index(node_v, base_adj.size(0), "node_v")
        base_adj = base_adj.to(device)
        h_before = self.calc_entropy(base_adj)

        perturbed_adj = base_adj.clone()
        perturbed_adj[node_u, node_v] = 1.0
        perturbed_adj[node_v, node_u] = 1.0

        h_after = self.calc_entropy(perturbed_adj)
        delta_h = float((h_after - h_before).item())

        p_base = F.softmax(base_adj, dim=-1)
        p_pert = F.softmax(perturbed_adj, dim=-1)
        # KL(p_pert ‖ p_base): 새 가설(perturbed)이 원래 분포(base)로부터
        # 얼마나 벗어났는지를 측정한다. F.kl_div(input, target)는
        # target * (log(target) - input)을 계산하므로, target=p_pert,
        # input=log(p_base)로 넣어야 KL(p_pert ‖ p_base)가 나온다.
        kl_div = float(F.kl_div(p_base.log(), p_pert, reduction='batchmean').item())

        if delta_h > self.entropy_tolerance or kl_div > self.divergence_limit:
            return False, delta_h, f"[REJECTED] 엔트로피 폭증 (Delta H: {delta_h:+.4f}, KL: {kl_div:.4f})"
        else:
            return True, delta_h, f"[VERIFIED] 구조적 안정 유지 (Delta H: {delta_h:+.4f}, KL: {kl_div:.4f})"

    def verify_candidates(
        self, base_adj: torch.Tensor, query_node_idx: int, candidate_doc_ids: List[int]
    ) -> List[VerificationResult]:
        """후보 문서 각각에 대해 (질의 노드, 문서 노드) 엣지 가설을 검증한다."""
        results = []
        for doc_id in candidate_doc_ids:
            is_valid, delta_h, log_msg = self.verify_hypothesis(base_adj, query_node_idx, doc_id)
            results.append(VerificationResult(doc_id=doc_id, verified=is_valid, delta_h=delta_h, log_message=log_msg))
        return results

    @staticmethod
    def _validate_adjacency(adj: torch.Tensor) -> None:
        if not isinstance(adj, torch.Tensor):
            raise TypeError("adjacency must be a torch.Tensor")
        if adj.ndim != 2 or adj.size(0) != adj.size(1):
            raise ValueError("adjacency must be a square matrix")
        if not torch.isfinite(adj).all().item():
            raise ValueError("adjacency must not contain NaN or Inf")

    @staticmethod
    def _validate_node_index(node: int, size: int, name: str) -> None:
        if not isinstance(node, int) or isinstance(node, bool):
            raise TypeError(f"{name} must be an integer")
        if not 0 <= node < size:
            raise ValueError(f"{name} must be between 0 and {size - 1}")

"""그래프(인접행렬) 상태 저장소.

인접행렬을 어떻게 만드는지(graph/builder.py)와 언제 롤백하는지
(safety/circuit_breaker.py)는 이 클래스가 모른다. 이 클래스는 오직
"현재 상태를 들고 있다가, 요청받으면 교체하거나 증강된 사본을 만들어준다"만
한다.
"""
from typing import List

import torch


class GraphStore:
    def __init__(self, adjacency: torch.Tensor) -> None:
        self._adjacency = adjacency

    def current(self) -> torch.Tensor:
        return self._adjacency

    def replace(self, adjacency: torch.Tensor) -> None:
        """인접행렬을 교체한다.

        (GPU-001 대응) 넘겨받은 텐서를 그대로 저장하지 않고, 이 GraphStore가
        원래 갖고 있던 device/dtype으로 맞춰서 저장한다. circuit breaker의
        rollback 경로(CRITICAL_ROLLBACK)를 통해 다른 device(예: GPU)에서
        계산된 텐서가 들어오더라도, GraphStore는 항상 자신의 canonical
        device/dtype을 유지한다는 불변식을 스스로 보장한다.
        (알고리즘 변경 아님 — 값은 바뀌지 않고 device/dtype만 맞춰짐.
        기존에 두 device가 이미 같았다면 이 연산은 항등(no-op)이다.)
        """
        self._adjacency = adjacency.to(device=self._adjacency.device, dtype=self._adjacency.dtype)

    @property
    def size(self) -> int:
        return self._adjacency.size(0)

    def augmented_with(self, edge_weights: List[float]) -> torch.Tensor:
        """새 노드 1개(마지막 인덱스)를 추가한 (n+1)x(n+1) 그래프를 반환한다.

        원본 인접행렬은 변경하지 않는다(새 텐서를 반환). 질의 노드를
        그래프에 임시로 얹어보는 용도로 쓰인다.
        (기존 hybrid_retriever._augmented_graph와 동일한 로직)
        """
        n = self.size
        aug = torch.eye(n + 1)
        aug[:n, :n] = self._adjacency
        for i, w in enumerate(edge_weights):
            aug[n, i] = w
            aug[i, n] = w
        return aug

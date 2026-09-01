"""재검색 전략.

confidence가 부족할 때 다음 RetrievalPlan을 어떻게 바꿀지 결정한다.
각 전략은 독립적으로 테스트 가능하도록 순수 함수에 가깝게 설계했다
(RetrievalPlan을 받아 새 RetrievalPlan을 반환).
"""
from typing import Protocol

from vram_engine.core.types import RetrievalPlan


class RetrievalStrategy(Protocol):
    def apply(self, plan: RetrievalPlan) -> RetrievalPlan:
        ...


class IncreaseSpectralWeightStrategy:
    """BM25 신호만으로 부족할 때 스펙트럴 비중을 높여 재검색한다."""

    def __init__(self, step: float = 0.2):
        self.step = step

    def apply(self, plan: RetrievalPlan) -> RetrievalPlan:
        new_alpha = max(0.0, plan.alpha - self.step)
        return RetrievalPlan(query_text=plan.query_text, alpha=new_alpha, top_k=plan.top_k)


class ExpandTopKStrategy:
    """후보 폭을 넓혀 재검색한다."""

    def __init__(self, factor: int = 2):
        self.factor = factor

    def apply(self, plan: RetrievalPlan) -> RetrievalPlan:
        return RetrievalPlan(query_text=plan.query_text, alpha=plan.alpha, top_k=plan.top_k * self.factor)

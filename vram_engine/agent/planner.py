"""검색 계획 수립.

initial_plan(): 첫 RetrievalPlan을 만든다.
should_retry(): confidence가 충분한지 판단한다.
next_plan(): 주어진 전략을 적용해 다음 RetrievalPlan을 만든다.

confidence_threshold=0.0으로 두면 항상 첫 결과를 채택한다 (재검색 없음).
기존 run_hybrid_demo.py와 동일한 단일 패스 동작을 보존하기 위해
interfaces/run_hybrid_demo.py의 호환 어댑터는 이 값을 0.0으로 사용한다.
"""
from vram_engine.agent.strategy import RetrievalStrategy
from vram_engine.core.types import RetrievalPlan


class Planner:
    def __init__(self, base_alpha: float = 0.5, base_top_k: int = 5, confidence_threshold: float = 0.6):
        self.base_alpha = base_alpha
        self.base_top_k = base_top_k
        self.confidence_threshold = confidence_threshold

    def initial_plan(self, query_text: str) -> RetrievalPlan:
        return RetrievalPlan(query_text=query_text, alpha=self.base_alpha, top_k=self.base_top_k)

    def should_retry(self, confidence: float, retries: int, max_retries: int) -> bool:
        return confidence < self.confidence_threshold and retries < max_retries

    def next_plan(self, prev_plan: RetrievalPlan, strategy: RetrievalStrategy) -> RetrievalPlan:
        return strategy.apply(prev_plan)

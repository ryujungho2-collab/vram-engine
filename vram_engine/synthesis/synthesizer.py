"""검증된 검색 결과 → 최종 결과(FinalAnswer) 변환.

지금 단계의 책임은 최소한이다: Agent가 넘겨준 (정렬된 후보, 검증 결과,
엔진 상태)를 그대로 FinalAnswer로 조립한다. 요약/재작성 같은 후처리는
아직 없다 — 그건 operator_registry.py의 TODO로 남겨둔 미래 확장 지점이다.
"""
from typing import List

from vram_engine.core.types import EngineStatus, FinalAnswer, RetrievalCandidate, VerificationResult


class Synthesizer:
    def synthesize(
        self,
        query_text: str,
        candidates: List[RetrievalCandidate],
        verifications: List[VerificationResult],
        engine_status: EngineStatus,
        retries: int = 0,
    ) -> FinalAnswer:
        return FinalAnswer(
            query_text=query_text,
            top_documents=candidates,
            verifications=verifications,
            engine_status=engine_status,
            retries=retries,
        )

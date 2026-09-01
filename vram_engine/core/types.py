"""공용 데이터 타입.

계층 간(Document Store → Graph Store → Retrieval → Agent → Verification →
Synthesis) 데이터 교환에 쓰이는 값 객체만 정의한다. 로직은 포함하지 않는다.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Document:
    doc_id: int
    text: str
    tokens: List[str]


@dataclass
class RetrievalCandidate:
    doc_id: int
    bm25_score: float
    spectral_score: float
    hybrid_score: float


@dataclass
class VerificationResult:
    doc_id: int
    verified: bool
    delta_h: float
    log_message: str


@dataclass
class EngineStatus:
    status: str  # "NORMAL" | "WARNING_PURGE" | "CRITICAL_ROLLBACK"
    entropy: float


@dataclass
class RetrievalPlan:
    query_text: str
    alpha: float
    top_k: int


@dataclass
class AgentStepResult:
    """Agent의 한 번의 Retrieve→Verify→Evaluate 사이클 기록."""
    plan: RetrievalPlan
    candidates: List[RetrievalCandidate]
    verifications: List[VerificationResult]
    confidence: float
    engine_status: EngineStatus


@dataclass
class FinalAnswer:
    query_text: str
    top_documents: List[RetrievalCandidate]
    verifications: List[VerificationResult]
    engine_status: EngineStatus
    retries: int = 0

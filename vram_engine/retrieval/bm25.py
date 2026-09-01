"""순수 어휘(lexical) 기반 BM25 검색기.

그래프/스펙트럴과 완전히 독립적이다 (ablation에서 "BM25 단독" 평가를
가능하게 하기 위한 핵심 분리 지점). 토크나이저는 core/document_store.py의
hybrid_tokenize를 그대로 재사용 — 여기서 별도로 정의하지 않는다.
"""
import math
from collections import Counter
from typing import Dict, List, Optional

from vram_engine.core.document_store import hybrid_tokenize
from vram_engine.core.types import Document


class SimpleBM25:
    """외부 의존성 없는 최소 Okapi BM25 구현. (알고리즘 변경 없음)"""

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens
        self.doc_freqs: List[Counter] = [Counter(doc) for doc in corpus_tokens]
        self.doc_lens = [len(doc) for doc in corpus_tokens]
        self.n_docs = len(corpus_tokens)
        self.avgdl = (sum(self.doc_lens) / self.n_docs) if self.n_docs else 0.0

        df: Counter = Counter()
        for doc in corpus_tokens:
            for term in set(doc):
                df[term] += 1
        self.idf: Dict[str, float] = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.n_docs
        for i in range(self.n_docs):
            doc_len = self.doc_lens[i] or 1
            freqs = self.doc_freqs[i]
            score = 0.0
            for term in query_tokens:
                if term not in freqs:
                    continue
                idf = self.idf.get(term, 0.0)
                f = freqs[term]
                denom = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * (f * (self.k1 + 1)) / denom
            scores[i] = score
        return scores


class BM25Retriever:
    """DocumentStore의 문서를 색인하고 질의 텍스트에 대한 BM25 원점수를 낸다."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._bm25: Optional[SimpleBM25] = None

    def index(self, documents: List[Document]) -> None:
        corpus_tokens = [doc.tokens for doc in documents]
        self._bm25 = SimpleBM25(corpus_tokens, k1=self.k1, b=self.b)

    def score(self, query_text: str) -> List[float]:
        if self._bm25 is None:
            raise RuntimeError("먼저 index(documents)를 호출하세요.")
        return self._bm25.get_scores(hybrid_tokenize(query_text))

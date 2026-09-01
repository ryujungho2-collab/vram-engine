"""문서 원본 저장소.

hybrid_tokenize()는 이 파일에 단 하나만 존재한다. retrieval/bm25.py와
graph/builder.py는 모두 이 함수를 그대로 import해서 쓴다 (예전에는
hybrid_retriever.py 안에 _tokenize로 중복될 위험이 있었음).
"""
from typing import List

from vram_engine.core.types import Document


def hybrid_tokenize(text: str) -> List[str]:
    """단어 토큰 + 문자 2-gram 하이브리드 토크나이저.

    영어처럼 공백으로 분리되는 언어는 단어 토큰이 그대로 잘 맞아떨어지지만,
    한국어 같은 교착어는 조사/어미 때문에 "양자역학에서" vs "양자"처럼 같은
    의미의 단어가 문자열 단위로는 전혀 겹치지 않는다. 문자 2-gram을 함께
    넣어주면 어미가 달라도 부분 문자열이 겹쳐 BM25/자카드 유사도가 0으로
    붕괴하는 것을 방지한다. (알고리즘 변경 없음 — 기존 구현 그대로 이식)
    """
    words = [t.lower() for t in text.split() if t.strip()]
    bigrams = [text[i:i + 2] for i in range(len(text) - 1) if text[i:i + 2].strip()]
    return words + bigrams


class DocumentStore:
    """문서 원본 + 토큰화 결과만 보유한다. 그래프/검색 로직은 모른다."""

    def __init__(self) -> None:
        self._documents: List[Document] = []

    def add(self, text: str) -> Document:
        doc = Document(doc_id=len(self._documents), text=text, tokens=hybrid_tokenize(text))
        self._documents.append(doc)
        return doc

    def add_many(self, texts: List[str]) -> List[Document]:
        return [self.add(t) for t in texts]

    def get(self, doc_id: int) -> Document:
        return self._documents[doc_id]

    def all(self) -> List[Document]:
        return list(self._documents)

    def __len__(self) -> int:
        return len(self._documents)

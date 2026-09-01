from vram_engine.core.document_store import DocumentStore, hybrid_tokenize
from vram_engine.retrieval.bm25 import BM25Retriever, SimpleBM25


def test_hybrid_tokenize_word_and_bigram():
    tokens = hybrid_tokenize("ab cd")
    assert "ab" in tokens
    assert "cd" in tokens
    assert "a" not in tokens  # 문자 2gram이므로 단일 문자는 없음


def test_bm25_scores_higher_for_matching_document():
    ds = DocumentStore()
    ds.add_many([
        "the quick brown fox jumps over the lazy dog",
        "completely unrelated sentence about cooking pasta",
    ])
    bm25 = BM25Retriever()
    bm25.index(ds.all())
    scores = bm25.score("quick brown fox")
    assert scores[0] > scores[1]


def test_korean_particle_regression_bm25_not_zero():
    """회귀 테스트: 한국어 조사/어미가 붙은 질의에서도 BM25가 0으로
    붕괴하지 않아야 한다 (하이브리드 토크나이저 덕분에 문자 2-gram이
    부분 매칭을 잡아준다)."""
    ds = DocumentStore()
    ds.add_many([
        "양자역학에서 파동함수의 붕괴는 관측 시점에 확률적으로 결정된다",
        "김치찌개는 돼지고기와 신 김치를 넣어 끓이는 한국 음식이다",
    ])
    bm25 = BM25Retriever()
    bm25.index(ds.all())
    scores = bm25.score("양자 확률 붕괴")
    assert scores[0] > 0.0, "조사가 다른 형태로 붙어있어도 BM25 점수가 0이면 안 된다"
    assert scores[0] > scores[1]


def test_simple_bm25_empty_query_returns_zero_scores():
    bm25 = SimpleBM25([["a", "b"], ["c", "d"]])
    scores = bm25.get_scores([])
    assert scores == [0.0, 0.0]

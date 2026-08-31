"""Tests for the standalone retrieve() seam, exercised with a fake vector store."""

from unittest.mock import patch

import pytest

from rag_core.config import RetrieverConfig
from rag_core.retriever.retrieve import retrieve


class _FakeDoc:
    def __init__(self, text, source="doc.md"):
        self.page_content = text
        self.metadata = {"source": source}


class _FakeVectorStore:
    def __init__(self, scored_results):
        self.scored_results = scored_results
        self.last_k = None

    def similarity_search_with_score(self, query, k):
        self.last_k = k
        return self.scored_results[:k]


def _scored(n):
    return [(_FakeDoc(f"text {i}", f"doc{i}.md"), 1.0 - i * 0.1) for i in range(n)]


def test_returns_scored_documents_best_first():
    store = _FakeVectorStore(_scored(5))
    cfg = RetrieverConfig(search_type="similarity", top_k=3)

    results = retrieve("question", store, cfg)

    assert len(results) == 3
    assert results[0][1] == 1.0
    assert results[-1][1] == pytest.approx(0.8)


def test_top_k_argument_overrides_config():
    store = _FakeVectorStore(_scored(5))
    cfg = RetrieverConfig(search_type="similarity", top_k=3)

    results = retrieve("question", store, cfg, top_k=2)

    assert len(results) == 2


def test_mmr_search_type_raises():
    store = _FakeVectorStore(_scored(5))
    cfg = RetrieverConfig(search_type="mmr", top_k=3)

    with pytest.raises(ValueError, match="MMR"):
        retrieve("question", store, cfg)


def test_no_reranker_fetches_exactly_top_k():
    store = _FakeVectorStore(_scored(10))
    cfg = RetrieverConfig(search_type="similarity", top_k=4)

    retrieve("question", store, cfg)

    assert store.last_k == 4


def test_reranker_widens_the_candidate_pool():
    store = _FakeVectorStore(_scored(20))
    cfg = RetrieverConfig(
        search_type="similarity", top_k=3, rerank={"provider": "cross_encoder", "fetch_k": 10}
    )

    class _FakeReranker:
        name = "fake"

        def rerank(self, query, documents, top_n):
            for i, doc in enumerate(documents[:top_n]):
                doc.metadata["rerank_score"] = 1.0 - i * 0.01
            return documents[:top_n]

    with patch("rag_core.retriever.retrieve.get_reranker", return_value=_FakeReranker()):
        results = retrieve("question", store, cfg)

    assert store.last_k == 10  # widened, not the requested top_k of 3
    assert len(results) == 3
    assert results[0][1] == 1.0

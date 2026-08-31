"""Tests for the retrieve()/retrieve_scored() seams, with a fake vector store."""

from unittest.mock import patch

import pytest
from rag_core.config import RetrieverConfig
from rag_core.retriever.retrieve import retrieve, retrieve_scored


class _FakeDoc:
    def __init__(self, text, source="doc.md"):
        self.page_content = text
        self.metadata = {"source": source}


class _FakeVectorStore:
    def __init__(self, scored_results):
        self.scored_results = scored_results
        self.last_k = None
        self.last_kwargs = None

    def similarity_search_with_relevance_scores(self, query, k, **kwargs):
        self.last_k = k
        self.last_kwargs = kwargs
        return self.scored_results[:k]


def _scored(n):
    return [(_FakeDoc(f"text {i}", f"doc{i}.md"), 1.0 - i * 0.1) for i in range(n)]


# --- retrieve_scored: the (document, score) contract ---


def test_scored_returns_pairs_best_first():
    store = _FakeVectorStore(_scored(5))
    cfg = RetrieverConfig(search_type="similarity", top_k=3)

    results = retrieve_scored("question", store, cfg)

    assert len(results) == 3
    assert results[0][1] == 1.0
    assert results[-1][1] == pytest.approx(0.8)


def test_scored_top_k_argument_overrides_config():
    store = _FakeVectorStore(_scored(5))
    cfg = RetrieverConfig(search_type="similarity", top_k=3)

    assert len(retrieve_scored("question", store, cfg, top_k=2)) == 2


def test_scored_raises_for_mmr():
    store = _FakeVectorStore(_scored(5))
    cfg = RetrieverConfig(search_type="mmr", top_k=3)

    with pytest.raises(ValueError, match="MMR"):
        retrieve_scored("question", store, cfg)


def test_no_reranker_fetches_exactly_top_k():
    store = _FakeVectorStore(_scored(10))
    cfg = RetrieverConfig(search_type="similarity", top_k=4)

    retrieve_scored("question", store, cfg)

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
        results = retrieve_scored("question", store, cfg)

    assert store.last_k == 10  # widened, not the requested top_k of 3
    assert len(results) == 3
    assert results[0][1] == 1.0


def test_filters_are_passed_through_untouched():
    store = _FakeVectorStore(_scored(5))
    cfg = RetrieverConfig(search_type="similarity", top_k=3)
    filters = {"source": {"$eq": "doc1.md"}}

    retrieve_scored("question", store, cfg, filters=filters)

    assert store.last_kwargs["filter"] == filters


# --- retrieve: plain documents, MMR-capable ---


def test_retrieve_returns_plain_documents():
    store = _FakeVectorStore(_scored(5))
    cfg = RetrieverConfig(search_type="similarity", top_k=3)

    results = retrieve("question", store, cfg)

    assert len(results) == 3
    # Documents, not (doc, score) tuples.
    assert all(hasattr(doc, "page_content") for doc in results)


def test_retrieve_stamps_score_into_metadata():
    store = _FakeVectorStore(_scored(3))
    cfg = RetrieverConfig(search_type="similarity", top_k=3)

    results = retrieve("question", store, cfg)

    assert results[0].metadata["score"] == 1.0
    assert results[1].metadata["score"] == pytest.approx(0.9)


def test_retrieve_supports_mmr_without_raising():
    store = _FakeVectorStore(_scored(5))
    cfg = RetrieverConfig(search_type="mmr", top_k=3)
    documents = [doc for doc, _ in _scored(3)]

    # MMR goes through hybrid.mmr_search (as_retriever().invoke()), which the
    # fake store does not implement — patch it to prove routing, not plumbing.
    with patch("rag_core.retriever.retrieve.mmr_search", return_value=documents) as mmr:
        results = retrieve("question", store, cfg)

    assert results == documents
    assert mmr.call_args.kwargs["k"] == 3

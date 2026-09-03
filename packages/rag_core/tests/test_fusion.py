"""RRFRetriever: reciprocal rank fusion over hand-built rankings (design_summary.md 3.2)."""

from langchain_core.documents import Document
from rag_core.retriever.base import SearchResult
from rag_core.retriever.fusion import RRFRetriever


class _FakeRetriever:
    """Returns a fixed, pre-ranked list regardless of query."""

    def __init__(self, doc_ids: list[str]):
        self._doc_ids = doc_ids

    def search(self, query: str, k: int | None = None) -> list[SearchResult]:
        ids = self._doc_ids if k is None else self._doc_ids[:k]
        return [
            SearchResult(
                doc_id=doc_id,
                document=Document(page_content="", metadata={"doc_id": doc_id}),
                score=float(len(ids) - i),
                score_type="bm25",
            )
            for i, doc_id in enumerate(ids)
        ]


def test_rrf_matches_hand_computed_scores():
    # child A ranks: d1, d2, d3
    # child B ranks: d2, d1, d4
    # d3 only appears in A, d4 only in B.
    a = _FakeRetriever(["d1", "d2", "d3"])
    b = _FakeRetriever(["d2", "d1", "d4"])
    retriever = RRFRetriever([a, b], rrf_k=60, top_k=10)

    results = retriever.search("query", k=10)
    scores = {r.doc_id: r.score for r in results}

    expected = {
        "d1": 1 / 61 + 1 / 62,  # rank 1 in A, rank 2 in B
        "d2": 1 / 62 + 1 / 61,  # rank 2 in A, rank 1 in B
        "d3": 1 / 63,  # rank 3 in A only
        "d4": 1 / 63,  # rank 3 in B only
    }
    assert scores.keys() == expected.keys()
    for doc_id, expected_score in expected.items():
        assert scores[doc_id] == expected_score

    assert all(r.score_type == "rrf" for r in results)
    # d1 and d2 tie; both outrank d3/d4 which also tie.
    assert set(r.doc_id for r in results[:2]) == {"d1", "d2"}
    assert set(r.doc_id for r in results[2:]) == {"d3", "d4"}


def test_rrf_requests_at_least_k_from_each_child():
    a = _FakeRetriever(["d1", "d2", "d3", "d4", "d5"])
    b = _FakeRetriever(["d6"])
    retriever = RRFRetriever([a, b], rrf_k=60, top_k=4)

    results = retriever.search("query", k=4)

    assert len(results) == 4
    assert results[0].doc_id == "d1"


def test_rrf_uses_default_top_k_when_k_not_given():
    a = _FakeRetriever(["d1", "d2"])
    retriever = RRFRetriever([a], rrf_k=60, top_k=1)

    results = retriever.search("query")

    assert len(results) == 1
    assert results[0].doc_id == "d1"

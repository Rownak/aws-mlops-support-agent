"""BiEncoderScorer and RerankingRetriever (design_summary.md 3.2, tasks 4.3-4.4)."""

import pytest
from langchain_core.documents import Document
from rag_core.retriever.base import SearchResult
from rag_core.retriever.rerank import BiEncoderScorer, RerankingRetriever


class _FakeEmbeddings:
    """Deterministic, dimension-2 embeddings so cosine ranking is checkable."""

    _VECTORS = {
        "cats are great": [1.0, 0.0],
        "dogs are great": [0.0, 1.0],
        "cats and dogs": [0.7, 0.7],
    }

    def embed_query(self, text: str) -> list[float]:
        return self._VECTORS[text]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._VECTORS[t] for t in texts]


def _result(doc_id: str, text: str) -> SearchResult:
    return SearchResult(
        doc_id=doc_id,
        document=Document(page_content=text, metadata={"doc_id": doc_id}),
        score=1.0,
        score_type="bm25",
    )


class _FakeRetriever:
    def __init__(self, results: list[SearchResult]):
        self._results = results

    def search(self, query: str, k: int | None = None) -> list[SearchResult]:
        return self._results if k is None else self._results[:k]


def test_bi_encoder_scorer_ranks_by_cosine_similarity():
    scorer = BiEncoderScorer(embeddings=_FakeEmbeddings())
    docs = [
        Document(page_content="dogs are great"),
        Document(page_content="cats are great"),
    ]

    ranked = scorer.rerank("cats are great", docs)

    assert ranked[0].page_content == "cats are great"
    assert ranked[0].metadata["rerank_score"] > ranked[1].metadata["rerank_score"]


def test_reranking_retriever_reorders_and_maps_doc_ids():
    inner = _FakeRetriever(
        [
            _result("d1", "dogs are great"),
            _result("d2", "cats are great"),
            _result("d3", "cats and dogs"),
        ]
    )
    scorer = BiEncoderScorer(embeddings=_FakeEmbeddings())
    retriever = RerankingRetriever(inner, scorer, candidate_k=3, top_k=2)

    results = retriever.search("cats are great", k=2)

    assert len(results) == 2
    assert results[0].doc_id == "d2"  # exact match, highest cosine
    assert all(r.score_type == "rerank_logit" for r in results)


def test_reranking_retriever_requests_candidate_k_from_inner():
    calls: list[int | None] = []

    class _RecordingRetriever:
        def search(self, query: str, k: int | None = None) -> list[SearchResult]:
            calls.append(k)
            return [_result("d1", "cats are great"), _result("d2", "dogs are great")]

    retriever = RerankingRetriever(
        _RecordingRetriever(), BiEncoderScorer(embeddings=_FakeEmbeddings()),
        candidate_k=50, top_k=10,
    )
    retriever.search("cats are great", k=10)

    assert calls == [50]


def test_reranking_retriever_rejects_candidate_k_below_top_k():
    with pytest.raises(ValueError, match="candidate_k"):
        RerankingRetriever(
            _FakeRetriever([]), BiEncoderScorer(embeddings=_FakeEmbeddings()),
            candidate_k=5, top_k=10,
        )

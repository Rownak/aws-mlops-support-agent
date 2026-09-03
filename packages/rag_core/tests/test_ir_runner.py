from dataclasses import dataclass

from langchain_core.documents import Document
from rag_core.evals.ir_runner import run_evaluation
from rag_core.retriever.base import SearchResult


@dataclass
class _Query:
    query_id: str
    text: str


class _FakeRetriever:
    """Returns a fixed ranking regardless of query text; k caps the result."""

    def __init__(self, ranking: list[str]):
        self._ranking = ranking

    def search(self, query: str, k: int) -> list[SearchResult]:
        return [
            SearchResult(
                doc_id=doc_id, document=Document(page_content=""), score=1.0, score_type="bm25"
            )
            for doc_id in self._ranking[:k]
        ]


QUERIES = {"q1": _Query("q1", "irrelevant text")}
QRELS = {"q1": {"d1": 2, "d3": 1}}


def test_run_evaluation_computes_requested_metrics():
    retriever = _FakeRetriever(["d1", "d2", "d3", "d4"])
    result = run_evaluation(
        retriever=retriever,
        queries=QUERIES,
        qrels=QRELS,
        experiment="fake",
        config={},
        dataset="toy",
        metrics=["ndcg@10", "recall@100", "mrr@10", "precision@10"],
    )

    assert set(result.metrics) == {"ndcg@10", "recall@100", "mrr@10", "precision@10"}
    assert result.metrics["recall@100"] == 1.0  # both relevant docs retrieved
    assert result.metrics["mrr@10"] == 1.0  # d1 is rank 1
    assert result.dataset == "toy"


def test_per_query_keeps_only_ndcg_at_10():
    retriever = _FakeRetriever(["d1", "d2", "d3", "d4"])
    result = run_evaluation(
        retriever=retriever,
        queries=QUERIES,
        qrels=QRELS,
        experiment="fake",
        config={},
        dataset="toy",
        metrics=["recall@100"],
    )

    assert len(result.per_query) == 1
    pq = result.per_query[0]
    assert pq.query_id == "q1"
    assert 0.0 < pq.ndcg_at_10 < 1.0  # d1 (gain 2) ranked first but d3 (gain 1) not second
    assert pq.retrieved == ["d1", "d2", "d3", "d4"]


def test_fetch_k_covers_the_widest_requested_metric():
    # recall@100 needs 100 candidates even though the retriever only has 4.
    retriever = _FakeRetriever(["d1", "d2", "d3", "d4"])
    result = run_evaluation(
        retriever=retriever,
        queries=QUERIES,
        qrels=QRELS,
        experiment="fake",
        config={},
        dataset="toy",
        metrics=["recall@100"],
    )
    assert result.metrics["recall@100"] == 1.0


def test_limit_caps_query_count():
    queries = {"q1": _Query("q1", "a"), "q2": _Query("q2", "b")}
    retriever = _FakeRetriever(["d1"])
    result = run_evaluation(
        retriever=retriever,
        queries=queries,
        qrels={},
        experiment="fake",
        config={},
        dataset="toy",
        metrics=["ndcg@10"],
        limit=1,
    )
    assert len(result.per_query) == 1

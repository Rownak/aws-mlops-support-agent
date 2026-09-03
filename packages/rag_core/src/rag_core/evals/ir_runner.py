"""Run a Retriever over a query set and score it against qrels.

Generalized from rag_bench_eval's phase-1/2 evaluator: takes plain
`queries`/`qrels` dicts and a `dataset` name so it carries no BEIR/NFCorpus-
specific dependency, and a `metrics` list so which scores get computed is a
config choice, not a code choice (design_summary.md build order, phase 3).
"""

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from rag_core.evals.metrics import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from rag_core.retriever.base import Retriever

_METRIC_FNS = {
    "ndcg@10": (ndcg_at_k, 10),
    "recall@100": (recall_at_k, 100),
    "mrr@10": (mrr_at_k, 10),
    "precision@10": (precision_at_k, 10),
}


@dataclass
class PerQueryResult:
    query_id: str
    ndcg_at_10: float
    latency_ms: int
    retrieved: list[str]


@dataclass
class RunResult:
    experiment: str
    dataset: str
    config: dict[str, Any]
    per_query: list[PerQueryResult]
    metrics: dict[str, float] = field(default_factory=dict)
    llm_calls: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def mean_ndcg_at_10(self) -> float:
        if not self.per_query:
            return 0.0
        return sum(r.ndcg_at_10 for r in self.per_query) / len(self.per_query)


def run_evaluation(
    retriever: Retriever,
    queries: dict[str, Any],
    qrels: dict[str, dict[str, int]],
    experiment: str,
    config: dict[str, Any],
    dataset: str,
    metrics: list[str] = ("ndcg@10",),
    limit: int | None = None,
) -> RunResult:
    """Search every query, score each ranking against `metrics`, and collect a RunResult.

    `queries` maps query_id -> an object with a `.text` attribute (matches
    both `rag_bench_eval`'s `Query` and a plain namespace). `limit` caps the
    number of queries — a fast smoke run rather than the full gate.
    """
    query_items = list(queries.items())[:limit] if limit else list(queries.items())
    fetch_k = max([_METRIC_FNS[name][1] for name in metrics] + [10])  # 10: per_query's ndcg@10

    per_query: list[PerQueryResult] = []
    metric_totals: dict[str, float] = {name: 0.0 for name in metrics}

    for query_id, query in query_items:
        start = time.perf_counter()
        results = retriever.search(query.text, fetch_k)
        latency_ms = int((time.perf_counter() - start) * 1000)

        retrieved = [r.doc_id for r in results]
        qrels_for_query = qrels.get(query_id, {})

        scores = {}
        for name in metrics:
            fn, k = _METRIC_FNS[name]
            scores[name] = fn(retrieved, qrels_for_query, k)
            metric_totals[name] += scores[name]

        per_query.append(
            PerQueryResult(
                query_id=query_id,
                ndcg_at_10=ndcg_at_k(retrieved, qrels_for_query, 10),
                latency_ms=latency_ms,
                retrieved=retrieved[:10],
            )
        )

    n = len(per_query)
    mean_metrics = {name: (total / n if n else 0.0) for name, total in metric_totals.items()}

    return RunResult(
        experiment=experiment,
        dataset=dataset,
        config=config,
        per_query=per_query,
        metrics=mean_metrics,
    )

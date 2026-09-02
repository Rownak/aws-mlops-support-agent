"""Run a Retriever over NFCorpus's queries and score it against qrels.

Runner stays local to rag_bench_eval for now — it moves to `rag_core.evals`
in phase 3 once a second retriever (dense) exists to generalize from
(design_summary.md build order).
"""

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_core.retriever.base import Retriever

from rag_bench_eval.datasets.types import Qrels, Query
from rag_bench_eval.metrics import ndcg_at_k
from rag_bench_eval.settings import RUNS_DIR

TOP_K = 10


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
    llm_calls: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def mean_ndcg_at_10(self) -> float:
        if not self.per_query:
            return 0.0
        return sum(r.ndcg_at_10 for r in self.per_query) / len(self.per_query)


def run_evaluation(
    retriever: Retriever,
    queries: dict[str, Query],
    qrels: Qrels,
    experiment: str,
    config: dict[str, Any],
    dataset: str = "nfcorpus",
    limit: int | None = None,
) -> RunResult:
    """Search every query, score each ranking's nDCG@10, and collect a RunResult.

    `limit` caps the number of queries — a fast smoke run rather than the
    full 323-query gate.
    """
    query_items = list(queries.items())[:limit] if limit else list(queries.items())

    per_query: list[PerQueryResult] = []
    for query_id, query in query_items:
        start = time.perf_counter()
        results = retriever.search(query.text, TOP_K)
        latency_ms = int((time.perf_counter() - start) * 1000)

        retrieved = [r.doc_id for r in results]
        score = ndcg_at_k(retrieved, qrels.get(query_id, {}), TOP_K)

        per_query.append(
            PerQueryResult(
                query_id=query_id,
                ndcg_at_10=score,
                latency_ms=latency_ms,
                retrieved=retrieved,
            )
        )

    return RunResult(
        experiment=experiment,
        dataset=dataset,
        config=config,
        per_query=per_query,
    )


def write_run_json(result: RunResult, runs_dir: Path = RUNS_DIR) -> Path:
    """Persist a RunResult to results/runs/<experiment>_<ts>.json.

    Only doc_ids and scores are written — `retrieved` is already doc_ids, and
    Document text never enters RunResult, so there is nothing to strip here
    (design_summary.md: document never reaches disk).
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    # Filesystem-safe timestamp: RunResult.timestamp is ISO-8601 with colons.
    ts = result.timestamp.replace(":", "-")
    path = runs_dir / f"{result.experiment}_{ts}.json"

    payload = {
        "experiment": result.experiment,
        "dataset": result.dataset,
        "timestamp": result.timestamp,
        "config": result.config,
        "metrics": {"ndcg@10": result.mean_ndcg_at_10},
        "per_query": [
            {
                "query_id": r.query_id,
                "ndcg@10": r.ndcg_at_10,
                "latency_ms": r.latency_ms,
                "retrieved": r.retrieved,
            }
            for r in result.per_query
        ],
        "llm_calls": result.llm_calls,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

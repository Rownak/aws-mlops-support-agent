"""Persist a rag_core.evals.ir_runner.RunResult to results/runs/*.json.

The runner itself lives in rag_core.evals.ir_runner (phase 3): two retrievers
now share it, so its generic shape (plain dicts, metric list from config)
moved out of rag_bench_eval. Only the run-JSON path convention
(`results/runs/`) is specific to this benchmark harness.
"""

import json
from pathlib import Path

from rag_core.evals.ir_runner import RunResult

from rag_bench_eval.settings import RUNS_DIR


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
        "metrics": result.metrics,
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

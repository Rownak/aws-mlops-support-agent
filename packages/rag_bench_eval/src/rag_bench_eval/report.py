"""Read results/runs/*.json -> a markdown comparison table, results/results.md.

One row per experiment: only the latest run (by timestamp) counts, so
re-running an experiment supersedes its old row instead of duplicating it.
"""

import json
from pathlib import Path

from rag_bench_eval.settings import RESULTS_DIR, RUNS_DIR

METRIC_COLUMNS = ["ndcg@10", "recall@100", "mrr@10", "precision@10"]


def _latest_run_per_experiment(runs_dir: Path = RUNS_DIR, dataset: str = "nfcorpus") -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for path in runs_dir.glob("*.json"):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["dataset"] != dataset:
            continue
        experiment = run["experiment"]
        if experiment not in latest or run["timestamp"] > latest[experiment]["timestamp"]:
            latest[experiment] = run
    return latest


def _mean_latency_ms(run: dict) -> float:
    per_query = run["per_query"]
    if not per_query:
        return 0.0
    return sum(q["latency_ms"] for q in per_query) / len(per_query)


def build_report(runs_dir: Path = RUNS_DIR, dataset: str = "nfcorpus") -> str:
    """Markdown table of the latest run per experiment, sorted by nDCG@10 descending."""
    runs = list(_latest_run_per_experiment(runs_dir, dataset).values())
    runs.sort(key=lambda run: run["metrics"].get("ndcg@10", 0.0), reverse=True)

    header = ["experiment", *METRIC_COLUMNS, "mean_latency_ms", "llm_calls"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for run in runs:
        metrics = run["metrics"]
        row = [
            run["experiment"],
            *(f"{metrics.get(col, 0.0):.4f}" for col in METRIC_COLUMNS),
            f"{_mean_latency_ms(run):.0f}",
            str(run["llm_calls"]),
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def write_report(
    runs_dir: Path = RUNS_DIR, results_dir: Path = RESULTS_DIR, dataset: str = "nfcorpus"
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if dataset == "nfcorpus" else f"_{dataset}"
    path = results_dir / f"results{suffix}.md"
    path.write_text(build_report(runs_dir, dataset), encoding="utf-8")
    return path

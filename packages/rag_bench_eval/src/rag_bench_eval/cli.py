"""rag-bench-eval CLI: uv run rag-bench-eval <download|list|run|report>

Pipelines come from benchmark.yaml; `run --all` sweeps the `sweep` list.
"""

import argparse

from rag_core.evals.ir_runner import run_evaluation

from rag_bench_eval.config import load_config
from rag_bench_eval.datasets.nfcorpus import download_nfcorpus, load_nfcorpus
from rag_bench_eval.evaluator import write_run_json
from rag_bench_eval.report import write_report
from rag_bench_eval.retrievers import build_pipeline_retriever


def _cmd_download(args: argparse.Namespace) -> None:
    download_nfcorpus(force=args.force)
    print("nfcorpus ready")


def _cmd_list(args: argparse.Namespace) -> None:
    config = load_config()
    pipelines = config["retrieval"]["pipelines"]
    sweep = config.get("sweep", [])
    for name in pipelines:
        marker = " (in sweep)" if name in sweep else ""
        print(f"{name}: {pipelines[name]['type']}{marker}")


def _run_one(experiment: str, config: dict, corpus: dict, queries: dict, qrels: dict, limit):
    pipelines = config["retrieval"]["pipelines"]
    if experiment not in pipelines:
        raise SystemExit(f"unknown experiment: {experiment!r} (available: {list(pipelines)})")

    pipeline_cfg = pipelines[experiment]
    retriever = build_pipeline_retriever(experiment, pipeline_cfg, config, corpus)
    metrics = config["evaluation"]["metrics"]

    result = run_evaluation(
        retriever=retriever,
        queries=queries,
        qrels=qrels,
        experiment=experiment,
        config=pipeline_cfg,
        dataset="nfcorpus",
        metrics=metrics,
        limit=limit,
    )
    path = write_run_json(result)

    n = len(result.per_query)
    scores = "  ".join(f"{name} = {score:.4f}" for name, score in result.metrics.items())
    print(f"{experiment}: {scores}  over {n} queries")
    print(f"wrote {path}")


def _cmd_report(args: argparse.Namespace) -> None:
    path = write_report()
    print(f"wrote {path}")


def _cmd_run(args: argparse.Namespace) -> None:
    config = load_config()
    corpus_docs, queries, qrels = load_nfcorpus()
    corpus = {doc_id: doc.content for doc_id, doc in corpus_docs.items()}

    if args.all:
        for experiment in config["sweep"]:
            _run_one(experiment, config, corpus, queries, qrels, args.limit)
        return

    if not args.experiment:
        raise SystemExit("run requires --experiment <name> or --all")
    _run_one(args.experiment, config, corpus, queries, qrels, args.limit)


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-bench-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_download = subparsers.add_parser("download", help="fetch and cache NFCorpus")
    p_download.add_argument("--force", action="store_true", help="re-fetch even if cached")
    p_download.set_defaults(func=_cmd_download)

    p_list = subparsers.add_parser("list", help="print pipelines available in benchmark.yaml")
    p_list.set_defaults(func=_cmd_list)

    p_run = subparsers.add_parser("run", help="run one or more experiments, writing results/runs/")
    p_run.add_argument("--experiment", help="pipeline name from benchmark.yaml")
    p_run.add_argument("--all", action="store_true", help="run every pipeline in the sweep list")
    p_run.add_argument("--limit", type=int, default=None, help="cap query count for a smoke run")
    p_run.set_defaults(func=_cmd_run)

    p_report = subparsers.add_parser("report", help="write results/results.md from results/runs/")
    p_report.set_defaults(func=_cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

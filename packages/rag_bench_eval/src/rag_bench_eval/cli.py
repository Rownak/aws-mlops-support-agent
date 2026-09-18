"""rag-bench-eval CLI: uv run rag-bench-eval <download|list|run|report>

Pipelines come from the dataset's own config file (`--dataset`, see
`DATASETS` below); `run --all` sweeps its `sweep` list.
"""

import argparse

from rag_core.evals.ir_runner import run_evaluation

from rag_bench_eval.config import CONFIG_PATH, load_config
from rag_bench_eval.datasets.cqadupstack_programmers import load_cqadupstack_programmers
from rag_bench_eval.datasets.nfcorpus import download_nfcorpus, load_nfcorpus
from rag_bench_eval.datasets.ragbench_emanual import load_ragbench_emanual
from rag_bench_eval.evaluator import write_run_json
from rag_bench_eval.report import write_report
from rag_bench_eval.retrievers import build_pipeline_retriever
from rag_bench_eval.settings import PACKAGE_ROOT

# --dataset name -> (loader, config path, dataset name recorded in run JSON,
# report caveat). Unknown names raise here rather than falling through to a
# default. caveat is None for datasets whose numbers need no qualifier.
DATASETS = {
    "nfcorpus": (load_nfcorpus, CONFIG_PATH, "nfcorpus", None),
    "cqadupstack_programmers": (
        load_cqadupstack_programmers,
        CONFIG_PATH,
        "cqadupstack_programmers",
        None,
    ),
    "emanual": (
        load_ragbench_emanual,
        PACKAGE_ROOT / "benchmark.emanual.yaml",
        "ragbench_emanual",
        "Qrels are derived from cited-sentence annotations, not human relevance "
        "judgments, and mark only what the annotated answer cited — an unlabeled "
        "document may still be relevant. Every metric here is a lower bound, not "
        "directly comparable to a BEIR dataset's (design.md §4.1/9.3).",
    ),
}


def _resolve_dataset(name: str) -> tuple[dict, dict, dict, dict, str]:
    if name not in DATASETS:
        raise SystemExit(f"unknown dataset: {name!r} (available: {list(DATASETS)})")
    loader, config_path, dataset_name, _caveat = DATASETS[name]
    config = load_config(config_path)
    corpus_docs, queries, qrels = loader()
    return config, corpus_docs, queries, qrels, dataset_name


def _cmd_download(args: argparse.Namespace) -> None:
    download_nfcorpus(force=args.force)
    print("nfcorpus ready")


def _cmd_list(args: argparse.Namespace) -> None:
    _, config_path, _, _ = DATASETS[args.dataset]
    config = load_config(config_path)
    pipelines = config["retrieval"]["pipelines"]
    sweep = config.get("sweep", [])
    for name in pipelines:
        marker = " (in sweep)" if name in sweep else ""
        print(f"{name}: {pipelines[name]['type']}{marker}")


def _run_one(
    experiment: str,
    config: dict,
    corpus: dict,
    queries: dict,
    qrels: dict,
    limit,
    dataset: str = "nfcorpus",
):
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
        dataset=dataset,
        metrics=metrics,
        limit=limit,
    )
    path = write_run_json(result)

    n = len(result.per_query)
    scores = "  ".join(f"{name} = {score:.4f}" for name, score in result.metrics.items())
    print(f"{experiment}: {scores}  over {n} queries")
    print(f"wrote {path}")


def _cmd_report(args: argparse.Namespace) -> None:
    if args.dataset not in DATASETS:
        raise SystemExit(f"unknown dataset: {args.dataset!r} (available: {list(DATASETS)})")
    _, _, dataset_name, caveat = DATASETS[args.dataset]
    path = write_report(dataset=dataset_name, caveat=caveat)
    print(f"wrote {path}")


def _cmd_run(args: argparse.Namespace) -> None:
    config, corpus_docs, queries, qrels, dataset_name = _resolve_dataset(args.dataset)
    corpus = {doc_id: doc.content for doc_id, doc in corpus_docs.items()}

    if args.all:
        for experiment in config["sweep"]:
            _run_one(experiment, config, corpus, queries, qrels, args.limit, dataset_name)
        return

    if not args.experiment:
        raise SystemExit("run requires --experiment <name> or --all")
    _run_one(args.experiment, config, corpus, queries, qrels, args.limit, dataset_name)


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-bench-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_download = subparsers.add_parser("download", help="fetch and cache NFCorpus")
    p_download.add_argument("--force", action="store_true", help="re-fetch even if cached")
    p_download.set_defaults(func=_cmd_download)

    p_list = subparsers.add_parser("list", help="print pipelines available for a dataset's config")
    p_list.add_argument(
        "--dataset", default="nfcorpus", choices=list(DATASETS), help="dataset whose config to list"
    )
    p_list.set_defaults(func=_cmd_list)

    p_run = subparsers.add_parser("run", help="run one or more experiments, writing results/runs/")
    p_run.add_argument("--experiment", help="pipeline name from the dataset's config")
    p_run.add_argument("--all", action="store_true", help="run every pipeline in the sweep list")
    p_run.add_argument(
        "--dataset", default="nfcorpus", choices=list(DATASETS), help="dataset to use"
    )
    p_run.add_argument("--limit", type=int, default=None, help="cap query count for a smoke run")
    p_run.set_defaults(func=_cmd_run)

    p_report = subparsers.add_parser("report", help="write results/results*.md from results/runs/")
    p_report.add_argument(
        "--dataset", default="nfcorpus", choices=list(DATASETS), help="dataset to report on"
    )
    p_report.set_defaults(func=_cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

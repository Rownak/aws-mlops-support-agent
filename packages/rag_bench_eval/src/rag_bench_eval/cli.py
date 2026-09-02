"""rag-bench-eval CLI: uv run rag-bench-eval <download|run>

Phase 1 supports one experiment (bm25); `benchmark.yaml`-driven pipelines and
`--all` sweeps land in phase 2+ once build_retriever() exists.
"""

import argparse

from rag_core.retriever.bm25 import BM25Retriever

from rag_bench_eval.datasets.nfcorpus import download_nfcorpus, load_nfcorpus
from rag_bench_eval.evaluator import run_evaluation, write_run_json

# Phase 1 defaults — move into benchmark.yaml once a second retriever exists.
BM25_K1 = 1.5
BM25_B = 0.75


def _cmd_download(args: argparse.Namespace) -> None:
    download_nfcorpus(force=args.force)
    print("nfcorpus ready")


def _cmd_run(args: argparse.Namespace) -> None:
    if args.experiment != "bm25":
        raise SystemExit(f"unknown experiment: {args.experiment!r} (phase 1 only has 'bm25')")

    corpus, queries, qrels = load_nfcorpus()
    config = {"type": "bm25", "k1": BM25_K1, "b": BM25_B, "top_k": 10}
    retriever = BM25Retriever(
        corpus={doc_id: doc.content for doc_id, doc in corpus.items()},
        k1=BM25_K1,
        b=BM25_B,
    )

    result = run_evaluation(
        retriever=retriever,
        queries=queries,
        qrels=qrels,
        experiment=args.experiment,
        config=config,
        limit=args.limit,
    )
    path = write_run_json(result)

    n = len(result.per_query)
    print(f"{args.experiment}: nDCG@10 = {result.mean_ndcg_at_10:.4f} over {n} queries")
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-bench-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_download = subparsers.add_parser("download", help="fetch and cache NFCorpus")
    p_download.add_argument("--force", action="store_true", help="re-fetch even if cached")
    p_download.set_defaults(func=_cmd_download)

    p_run = subparsers.add_parser("run", help="run one experiment and write results/runs/*.json")
    p_run.add_argument("--experiment", required=True, help="experiment name, e.g. bm25")
    p_run.add_argument("--limit", type=int, default=None, help="cap query count for a smoke run")
    p_run.set_defaults(func=_cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

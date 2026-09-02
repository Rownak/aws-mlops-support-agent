"""rag-bench-eval CLI: uv run rag-bench-eval <download|run>

Phase 2 reads pipeline settings from benchmark.yaml and builds bm25/dense
directly; the recursive build_retriever() dispatch (for the nested pipelines
in later phases) lands in 2.7.
"""

import argparse

from rag_core.retriever.bm25 import BM25Retriever
from rag_core.retriever.dense import DenseRetriever

from rag_bench_eval.config import load_config
from rag_bench_eval.datasets.nfcorpus import download_nfcorpus, load_nfcorpus
from rag_bench_eval.evaluator import run_evaluation, write_run_json
from rag_bench_eval.index_cache import get_or_build
from rag_bench_eval.resources import get_embeddings


def _cmd_download(args: argparse.Namespace) -> None:
    download_nfcorpus(force=args.force)
    print("nfcorpus ready")


def _build_retriever(name: str, pipeline_cfg: dict, config: dict, corpus: dict[str, str]):
    ptype = pipeline_cfg["type"]

    if ptype == "bm25":
        k1, b = pipeline_cfg["k1"], pipeline_cfg["b"]
        return get_or_build(
            ("bm25", k1, b),
            lambda: BM25Retriever(corpus=corpus, k1=k1, b=b, top_k=pipeline_cfg["top_k"]),
        )

    if ptype == "dense":
        embeddings_name = pipeline_cfg["embeddings"]
        metric = pipeline_cfg["metric"]
        embeddings = get_embeddings(embeddings_name, config)
        return get_or_build(
            ("dense", embeddings_name, metric),
            lambda: DenseRetriever(
                corpus=corpus, embeddings=embeddings, top_k=pipeline_cfg["top_k"]
            ),
        )

    raise ValueError(f"unknown pipeline type: {ptype!r} (pipeline {name!r})")


def _cmd_run(args: argparse.Namespace) -> None:
    config = load_config()
    pipelines = config["retrieval"]["pipelines"]
    if args.experiment not in pipelines:
        raise SystemExit(
            f"unknown experiment: {args.experiment!r} (available: {list(pipelines)})"
        )

    corpus_docs, queries, qrels = load_nfcorpus()
    corpus = {doc_id: doc.content for doc_id, doc in corpus_docs.items()}

    pipeline_cfg = pipelines[args.experiment]
    retriever = _build_retriever(args.experiment, pipeline_cfg, config, corpus)

    result = run_evaluation(
        retriever=retriever,
        queries=queries,
        qrels=qrels,
        experiment=args.experiment,
        config=pipeline_cfg,
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
    p_run.add_argument("--experiment", required=True, help="pipeline name from benchmark.yaml")
    p_run.add_argument("--limit", type=int, default=None, help="cap query count for a smoke run")
    p_run.set_defaults(func=_cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

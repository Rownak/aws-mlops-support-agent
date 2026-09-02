"""BEIR/NFCorpus: download (via Hugging Face `datasets`), cache, and parse.

Download and parse live together deliberately (design_summary.md): a second
BEIR dataset is what would reveal which parts of this are format-generic vs.
NFCorpus-specific, and guessing that split now would likely guess wrong.

Hugging Face hosts BEIR/NFCorpus as two repos:
- ``BeIR/nfcorpus``       configs "corpus" and "queries", each one split
- ``BeIR/nfcorpus-qrels`` split "test" (also "train"/"validation", unused here)

``load_dataset`` caches the download under HF's own cache dir (`HF_HOME`, not
`NFCORPUS_DIR`); the manifest below just records what version we last pulled
and how many rows it had, for a quick "did this change" check.
"""

import json
import logging

from datasets import load_dataset

from rag_bench_eval.datasets.types import Doc, Qrels, Query
from rag_bench_eval.settings import NFCORPUS_DIR

logger = logging.getLogger(__name__)

CORPUS_REPO = "BeIR/nfcorpus"
QRELS_REPO = "BeIR/nfcorpus-qrels"
MANIFEST_NAME = ".manifest.json"


def download_nfcorpus(force: bool = False) -> None:
    """Pull NFCorpus (corpus, queries, test qrels) via `datasets` and cache a manifest.

    `load_dataset` itself is idempotent (HF caches by repo+config+split), so
    `force` just re-derives and rewrites the manifest rather than re-downloading.
    """
    manifest_path = NFCORPUS_DIR / MANIFEST_NAME
    if manifest_path.exists() and not force:
        logger.info(f"nfcorpus: manifest found at {manifest_path}, skipping")
        return

    corpus = load_dataset(CORPUS_REPO, "corpus", split="corpus")
    queries = load_dataset(CORPUS_REPO, "queries", split="queries")
    qrels_test = load_dataset(QRELS_REPO, split="test")

    NFCORPUS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "corpus_repo": CORPUS_REPO,
        "qrels_repo": QRELS_REPO,
        "counts": {
            "corpus": len(corpus),
            "queries": len(queries),
            "qrels_test": len(qrels_test),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"nfcorpus: ready ({manifest['counts']})")


def load_nfcorpus() -> tuple[dict[str, Doc], dict[str, Query], Qrels]:
    """Load the test split: (corpus by doc_id, queries by query_id, qrels).

    Assumes download_nfcorpus() has already run (HF's own cache makes this
    fast even on a fresh process).
    """
    corpus_rows = load_dataset(CORPUS_REPO, "corpus", split="corpus")
    corpus = {
        row["_id"]: Doc(doc_id=row["_id"], title=row["title"], text=row["text"])
        for row in corpus_rows
    }

    query_rows = load_dataset(CORPUS_REPO, "queries", split="queries")
    all_queries = {row["_id"]: Query(query_id=row["_id"], text=row["text"]) for row in query_rows}

    qrels_rows = load_dataset(QRELS_REPO, split="test")
    qrels: Qrels = {}
    for row in qrels_rows:
        query_id, doc_id, score = row["query-id"], row["corpus-id"], int(row["score"])
        qrels.setdefault(query_id, {})[doc_id] = score

    # Test split only: restrict queries to those that actually have qrels.
    queries = {qid: q for qid, q in all_queries.items() if qid in qrels}

    return corpus, queries, qrels

"""BEIR/CQADupStack Programmers: parse from local BEIR-format files.

Unlike NFCorpus, this subset isn't on Hugging Face `datasets` — the files
were downloaded ahead of time to data/cqadupstack/programmers/ in the
standard BEIR layout: corpus.jsonl, queries.jsonl, qrels/test.tsv.
"""

import json
import logging

from rag_bench_eval.datasets.types import Doc, Qrels, Query
from rag_bench_eval.settings import CQADUPSTACK_PROGRAMMERS_DIR

logger = logging.getLogger(__name__)

MANIFEST_NAME = ".manifest.json"


def load_cqadupstack_programmers() -> tuple[dict[str, Doc], dict[str, Query], Qrels]:
    """Load the test split: (corpus by doc_id, queries by query_id, qrels).

    Assumes the BEIR-format files already exist locally under
    CQADUPSTACK_PROGRAMMERS_DIR (corpus.jsonl, queries.jsonl, qrels/test.tsv).
    """
    corpus: dict[str, Doc] = {}
    with open(CQADUPSTACK_PROGRAMMERS_DIR / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[row["_id"]] = Doc(
                doc_id=row["_id"], title=row.get("title", ""), text=row["text"]
            )

    all_queries: dict[str, Query] = {}
    with open(CQADUPSTACK_PROGRAMMERS_DIR / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            all_queries[row["_id"]] = Query(query_id=row["_id"], text=row["text"])

    qrels: Qrels = {}
    with open(CQADUPSTACK_PROGRAMMERS_DIR / "qrels" / "test.tsv", encoding="utf-8") as f:
        f.readline()  # header: query-id, corpus-id, score
        for line in f:
            query_id, doc_id, score = line.rstrip("\n").split("\t")
            qrels.setdefault(query_id, {})[doc_id] = int(score)

    # Test split only: restrict queries to those that actually have qrels.
    queries = {qid: q for qid, q in all_queries.items() if qid in qrels}

    manifest = {
        "counts": {
            "corpus": len(corpus),
            "queries": len(queries),
            "qrels_test": len(qrels),
        },
    }
    manifest_path = CQADUPSTACK_PROGRAMMERS_DIR / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"cqadupstack_programmers: loaded ({manifest['counts']})")

    return corpus, queries, qrels

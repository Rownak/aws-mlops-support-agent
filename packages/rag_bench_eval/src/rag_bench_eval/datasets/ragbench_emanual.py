"""RAGBench/emanual: pool per-row inline documents into one corpus, derive qrels.

Unlike BEIR, emanual ships one row per question with its own documents inline
and no qrels table (design.md §4.1). A document relevant to one row is a
distractor for the rest, so all rows' documents are pooled into a single
corpus — that pooling is what makes retrieval a real task here rather than
scoring against an ~100%-relevant list.

Relevance comes from `all_relevant_sentence_keys`, which key into
`documents_sentences` (a per-row list of per-document `[key, sentence]`
pairs). Mapping a key back to its document gives a derived, binary qrels
label — not the graded human judgment BEIR datasets carry.
"""

import hashlib
import logging

from datasets import load_dataset

from rag_bench_eval.datasets.types import Doc, Qrels, Query

logger = logging.getLogger(__name__)

REPO = "galileo-ai/ragbench"
CONFIG = "emanual"


def _doc_id(text: str) -> str:
    """Stable id from content, not row position — must not shift on filter/reorder."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _relevant_doc_ids(row: dict, texts_by_index: list[str]) -> set[str]:
    """Map this row's relevant sentence keys back to the pooled doc_ids they came from."""
    # sentence key (e.g. "2e") -> index of the document it belongs to in this row
    doc_index_by_key: dict[str, int] = {}
    for doc_index, sentences in enumerate(row["documents_sentences"]):
        for key, _sentence in sentences:
            doc_index_by_key[key] = doc_index

    relevant: set[str] = set()
    for raw_key in row["all_relevant_sentence_keys"]:
        # A handful of keys carry a trailing "." not present in documents_sentences.
        key = raw_key.rstrip(".")
        doc_index = doc_index_by_key[key]
        relevant.add(_doc_id(texts_by_index[doc_index]))
    return relevant


def load_ragbench_emanual() -> tuple[dict[str, Doc], dict[str, Query], Qrels]:
    """Load the test split: (pooled corpus by doc_id, queries by query_id, derived qrels).

    The test split carries each question twice under the same `id` — two
    independent annotation passes over the same question/documents, each with
    its own `all_relevant_sentence_keys`. Both passes' relevant documents are
    unioned into one qrels entry per id, rather than letting the second row
    silently overwrite the first.
    """
    rows = load_dataset(REPO, CONFIG, split="test")

    corpus: dict[str, Doc] = {}
    queries: dict[str, Query] = {}
    relevant_by_query: dict[str, set[str]] = {}
    empty_rows = 0

    for row in rows:
        texts_by_index = list(row["documents"])
        for text in texts_by_index:
            doc_id = _doc_id(text)
            if doc_id not in corpus:
                corpus[doc_id] = Doc(doc_id=doc_id, title="", text=text)

        relevant = _relevant_doc_ids(row, texts_by_index)
        if not relevant:
            empty_rows += 1

        query_id = row["id"]
        queries.setdefault(query_id, Query(query_id=query_id, text=row["question"]))
        relevant_by_query.setdefault(query_id, set()).update(relevant)

    # Drop queries with no relevant documents from either annotation pass
    # (unanswerable questions) rather than scoring them against an empty qrels entry.
    qrels: Qrels = {
        qid: {doc_id: 1 for doc_id in relevant}
        for qid, relevant in relevant_by_query.items()
        if relevant
    }
    queries = {qid: q for qid, q in queries.items() if qid in qrels}
    dropped = len(relevant_by_query) - len(qrels)

    logger.info(
        f"ragbench_emanual: loaded (corpus={len(corpus)}, queries={len(queries)}, "
        f"dropped_unanswerable={dropped}, empty_annotation_passes={empty_rows})"
    )
    return corpus, queries, qrels

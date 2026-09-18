"""ragbench_emanual: corpus pooling/dedup (8.3) and qrels derivation (8.4).

The pooling/dedup test hits the network (loads the real 132-row test split).
The qrels-derivation test builds a two-row fixture inline — no network,
matching types.py's rule that dataset shape can be exercised without a
download.
"""

from rag_bench_eval.datasets import ragbench_emanual
from rag_bench_eval.datasets.ragbench_emanual import (
    _doc_id,
    _relevant_doc_ids,
    load_ragbench_emanual,
)


def test_load_ragbench_emanual_pools_and_dedups():
    corpus, queries, qrels = load_ragbench_emanual()

    # The test split carries each question twice under the same id (two
    # annotation passes) - 132 rows collapse to at most 66 unique queries.
    assert len(queries) <= 66
    assert len(queries) > 0
    # Pooling across rows: bigger than any single row's document list (max a
    # handful), proof the corpus isn't just the last row's documents.
    assert len(corpus) > 10
    assert all(qid in qrels for qid in queries)
    assert all(doc.title == "" for doc in corpus.values())


def test_relevant_doc_ids_maps_sentence_keys_to_documents():
    row = {
        "documents": ["doc zero text", "doc one text"],
        "documents_sentences": [
            [["0a", "doc zero text"]],
            [["1a", "doc one"], ["1b", "text"]],
        ],
        "all_relevant_sentence_keys": ["1b"],
    }
    texts_by_index = row["documents"]

    relevant = _relevant_doc_ids(row, texts_by_index)

    assert relevant == {_doc_id("doc one text")}


def test_relevant_doc_ids_strips_trailing_dot_in_key():
    row = {
        "documents": ["only doc"],
        "documents_sentences": [[["0a", "only doc"]]],
        "all_relevant_sentence_keys": ["0a."],
    }

    relevant = _relevant_doc_ids(row, row["documents"])

    assert relevant == {_doc_id("only doc")}


def test_two_row_fixture_shares_one_corpus_entry_for_duplicate_text():
    """A text appearing in two rows must yield one corpus entry, keyed by content."""
    shared_text = "shared manual passage"
    rows = [
        {
            "id": "q1",
            "question": "question one",
            "documents": [shared_text, "row one only"],
            "documents_sentences": [
                [["0a", shared_text]],
                [["1a", "row one only"]],
            ],
            "all_relevant_sentence_keys": ["0a"],
        },
        {
            "id": "q2",
            "question": "question two",
            "documents": [shared_text],
            "documents_sentences": [[["0a", shared_text]]],
            "all_relevant_sentence_keys": ["0a"],
        },
    ]

    corpus = {}
    queries = {}
    qrels = {}
    for row in rows:
        texts_by_index = row["documents"]
        for text in texts_by_index:
            doc_id = _doc_id(text)
            corpus.setdefault(doc_id, text)
        relevant = _relevant_doc_ids(row, texts_by_index)
        queries[row["id"]] = row["question"]
        qrels[row["id"]] = {doc_id: 1 for doc_id in relevant}

    assert len(corpus) == 2  # shared_text counted once, "row one only" once
    assert qrels == {
        "q1": {_doc_id(shared_text): 1},
        "q2": {_doc_id(shared_text): 1},
    }


def test_duplicate_row_ids_union_relevant_docs_across_annotation_passes(monkeypatch):
    """The real dataset carries each question twice under one id (two
    annotation passes with different relevant-key sets) - both passes'
    relevant docs must be unioned, not overwritten (no network: fake rows)."""
    doc_a, doc_b = "manual passage a", "manual passage b"
    fake_rows = [
        {
            "id": "emanual_1",
            "question": "how do I do X",
            "documents": [doc_a, doc_b],
            "documents_sentences": [[["0a", doc_a]], [["1a", doc_b]]],
            "all_relevant_sentence_keys": ["0a"],  # pass 1: only doc_a
        },
        {
            "id": "emanual_1",
            "question": "how do I do X",
            "documents": [doc_a, doc_b],
            "documents_sentences": [[["0a", doc_a]], [["1a", doc_b]]],
            "all_relevant_sentence_keys": ["1a"],  # pass 2: only doc_b
        },
    ]
    monkeypatch.setattr(ragbench_emanual, "load_dataset", lambda repo, config, split: fake_rows)

    corpus, queries, qrels = load_ragbench_emanual()

    assert len(queries) == 1
    assert qrels["emanual_1"] == {_doc_id(doc_a): 1, _doc_id(doc_b): 1}


def test_row_with_no_relevant_keys_in_either_pass_is_dropped(monkeypatch):
    fake_rows = [
        {
            "id": "emanual_2",
            "question": "unanswerable question",
            "documents": ["some doc"],
            "documents_sentences": [[["0a", "some doc"]]],
            "all_relevant_sentence_keys": [],
        },
        {
            "id": "emanual_2",
            "question": "unanswerable question",
            "documents": ["some doc"],
            "documents_sentences": [[["0a", "some doc"]]],
            "all_relevant_sentence_keys": [],
        },
    ]
    monkeypatch.setattr(ragbench_emanual, "load_dataset", lambda repo, config, split: fake_rows)

    corpus, queries, qrels = load_ragbench_emanual()

    assert queries == {}
    assert qrels == {}

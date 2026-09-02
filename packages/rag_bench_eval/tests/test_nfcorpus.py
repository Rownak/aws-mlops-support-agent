from rag_bench_eval.datasets.nfcorpus import download_nfcorpus, load_nfcorpus


def test_load_nfcorpus_counts():
    download_nfcorpus()
    corpus, queries, qrels = load_nfcorpus()

    assert len(corpus) == 3633
    assert len(queries) == 323
    assert len(qrels) > 0
    # Every query in the test split has at least one graded qrel.
    assert all(qid in qrels for qid in queries)

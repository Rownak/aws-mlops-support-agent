import math

from rag_bench_eval.metrics import ndcg_at_k

# Toy 5-doc ranking. Qrels grade three docs; two returned docs are unjudged (0).
QRELS = {"d1": 2, "d3": 1, "d5": 0}
RANKING = ["d1", "d2", "d3", "d4", "d5"]


def test_ndcg_hand_computed():
    # DCG = 2/log2(2) + 0/log2(3) + 1/log2(4) + 0/log2(5) + 0/log2(6)
    #     = 2/1 + 1/2 = 2.5
    dcg = 2 / math.log2(2) + 1 / math.log2(4)
    # Ideal ranking by gain: [2, 1, 0] -> IDCG = 2/log2(2) + 1/log2(3)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    expected = dcg / idcg

    assert math.isclose(ndcg_at_k(RANKING, QRELS, k=5), expected, rel_tol=1e-9)


def test_ndcg_perfect_ranking_is_one():
    perfect = ["d1", "d3", "d5", "d2", "d4"]
    assert math.isclose(ndcg_at_k(perfect, QRELS, k=5), 1.0, rel_tol=1e-9)


def test_ndcg_no_relevant_docs_returns_zero():
    assert ndcg_at_k(["x", "y"], {}, k=5) == 0.0


def test_ndcg_truncates_to_k():
    # Only the top-1 slot considered: d1 (gain 2) is both actual and ideal top gain.
    assert math.isclose(ndcg_at_k(RANKING, QRELS, k=1), 1.0, rel_tol=1e-9)


def test_ndcg_unjudged_docs_count_as_zero():
    ranking = ["unjudged1", "unjudged2"]
    assert ndcg_at_k(ranking, QRELS, k=5) == 0.0

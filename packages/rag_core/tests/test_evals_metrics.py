import math

from rag_core.evals.metrics import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k

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


def test_recall_hand_computed():
    # Relevant (grade > 0): d1, d3. Both appear in top 5.
    assert recall_at_k(RANKING, QRELS, k=5) == 1.0


def test_recall_partial():
    # Only d1 is within top 1; d3 is not.
    assert recall_at_k(RANKING, QRELS, k=1) == 0.5


def test_recall_no_relevant_docs_returns_zero():
    assert recall_at_k(["x", "y"], {}, k=5) == 0.0


def test_mrr_hand_computed():
    # First relevant doc (d1, grade 2) is at rank 1 -> RR = 1/1 = 1.0
    assert mrr_at_k(RANKING, QRELS, k=5) == 1.0


def test_mrr_first_relevant_at_rank_three():
    ranking = ["x", "y", "d3", "d1"]
    # First relevant doc (d3, grade 1) is at rank 3 -> RR = 1/3
    assert math.isclose(mrr_at_k(ranking, QRELS, k=5), 1 / 3, rel_tol=1e-9)


def test_mrr_no_relevant_within_k_returns_zero():
    assert mrr_at_k(["x", "y"], QRELS, k=2) == 0.0


def test_precision_hand_computed():
    # Top 5: d1(rel), d2(unjudged), d3(rel), d4(unjudged), d5(grade 0) -> 2/5
    assert math.isclose(precision_at_k(RANKING, QRELS, k=5), 2 / 5, rel_tol=1e-9)


def test_precision_truncates_to_k():
    # Top 1: d1 is relevant -> 1/1
    assert precision_at_k(RANKING, QRELS, k=1) == 1.0


def test_precision_empty_ranking_returns_zero():
    assert precision_at_k([], QRELS, k=5) == 0.0

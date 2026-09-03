"""IR metrics over ranked doc_ids and graded qrels: nDCG, recall, MRR, precision.

Corpus-agnostic: takes plain dicts, no rag_bench_eval dataset dependency.
"""

import math


def ndcg_at_k(ranked_doc_ids: list[str], qrels_for_query: dict[str, int], k: int) -> float:
    """Normalized discounted cumulative gain at rank k.

    `qrels_for_query` maps doc_id -> graded relevance (0/1/2). Doc_ids with no
    entry (unjudged) count as relevance 0, matching standard BEIR/trec_eval
    behaviour.
    """
    top_k = ranked_doc_ids[:k]

    dcg = sum(
        qrels_for_query.get(doc_id, 0) / math.log2(rank + 2)  # rank is 0-indexed
        for rank, doc_id in enumerate(top_k)
    )

    ideal_gains = sorted(qrels_for_query.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(ideal_gains))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def recall_at_k(ranked_doc_ids: list[str], qrels_for_query: dict[str, int], k: int) -> float:
    """Fraction of relevant docs (grade > 0) found in the top k."""
    relevant = {doc_id for doc_id, grade in qrels_for_query.items() if grade > 0}
    if not relevant:
        return 0.0
    top_k = set(ranked_doc_ids[:k])
    return len(relevant & top_k) / len(relevant)


def mrr_at_k(ranked_doc_ids: list[str], qrels_for_query: dict[str, int], k: int) -> float:
    """Reciprocal rank of the first relevant doc (grade > 0) within the top k."""
    for rank, doc_id in enumerate(ranked_doc_ids[:k]):
        if qrels_for_query.get(doc_id, 0) > 0:
            return 1.0 / (rank + 1)
    return 0.0


def precision_at_k(ranked_doc_ids: list[str], qrels_for_query: dict[str, int], k: int) -> float:
    """Fraction of the top k that are relevant (grade > 0)."""
    top_k = ranked_doc_ids[:k]
    if not top_k:
        return 0.0
    relevant_count = sum(1 for doc_id in top_k if qrels_for_query.get(doc_id, 0) > 0)
    return relevant_count / len(top_k)


def mean_ndcg_at_k(
    per_query_ranked: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int
) -> dict[str, float]:
    """ndcg_at_k for every query in `per_query_ranked`; returns {query_id: score}."""
    return {
        query_id: ndcg_at_k(ranked, qrels.get(query_id, {}), k)
        for query_id, ranked in per_query_ranked.items()
    }

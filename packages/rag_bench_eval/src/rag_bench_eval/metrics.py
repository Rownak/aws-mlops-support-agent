"""nDCG@k over ranked doc_ids and graded qrels.

Local to rag_bench_eval for now; moves to `rag_core.evals` in phase 3 once
dense also needs it (design_summary.md build order).
"""

import math

from rag_bench_eval.datasets.types import Qrels


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


def mean_ndcg_at_k(
    per_query_ranked: dict[str, list[str]], qrels: Qrels, k: int
) -> dict[str, float]:
    """ndcg_at_k for every query in `per_query_ranked`; returns {query_id: score}."""
    return {
        query_id: ndcg_at_k(ranked, qrels.get(query_id, {}), k)
        for query_id, ranked in per_query_ranked.items()
    }

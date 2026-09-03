"""Reciprocal rank fusion over n retrievers (design_summary.md 3.2).

RRF merges ranked lists without needing their scores to be on the same
scale — BM25 is unbounded, cosine is [-1, 1] — by scoring rank position
alone: `score = sum(1 / (rrf_k + rank))` over every child ranking a doc_id
appears in, ranks 1-based. A doc found by only one child still gets a
score, just a smaller one.
"""

from rag_core.retriever.base import Retriever, SearchResult


class RRFRetriever:
    """Fuses several retrievers' rankings by reciprocal rank."""

    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60, top_k: int = 10):
        self._retrievers = retrievers
        self.rrf_k = rrf_k
        self.top_k = top_k

    def search(self, query: str, k: int | None = None) -> list[SearchResult]:
        k = k if k is not None else self.top_k

        # Each child is asked for at least k results — fusing children that
        # each return only k could otherwise yield fewer than k fused hits
        # (design_summary.md's depth rule: a parent asks for the depth it needs).
        child_results = [retriever.search(query, k=k) for retriever in self._retrievers]

        scores: dict[str, float] = {}
        best_result: dict[str, SearchResult] = {}
        for results in child_results:
            for rank, result in enumerate(results, start=1):
                scores[result.doc_id] = scores.get(result.doc_id, 0.0) + 1.0 / (
                    self.rrf_k + rank
                )
                # Keep the first (highest-ranked) SearchResult seen for this
                # doc_id, for whatever document/metadata it carries.
                best_result.setdefault(result.doc_id, result)

        ranked_ids = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)[:k]

        return [
            SearchResult(
                doc_id=doc_id,
                document=best_result[doc_id].document,
                score=scores[doc_id],
                score_type="rrf",
            )
            for doc_id in ranked_ids
        ]

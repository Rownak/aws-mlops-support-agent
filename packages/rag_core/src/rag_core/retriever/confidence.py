"""Explainable retrieval-confidence heuristic.

Drives escalation in an agent: if retrieval looks weak, the agent should say
so (or offer a human hand-off) instead of bluffing an answer.

The heuristic (deliberately simple):
  1. No chunks retrieved                 -> not confident.
  2. Best score < min_top_score          -> not confident ("best match is weak").
  3. Otherwise                           -> confident.

`score_gap` (top1 - top2) is computed and reported but does NOT affect the
decision — it is surfaced so it can be watched on real queries (a big gap
suggests one clearly-best doc; a flat top-k can mean the query matched
everything a little and nothing well).

Cosine similarity is not a probability: a usable threshold depends on the
embedding model AND the corpus, which is why `min_top_score` is a caller-
supplied parameter rather than a constant here.
"""

from dataclasses import dataclass
from typing import Any

DEFAULT_MIN_TOP_SCORE = 0.35


@dataclass(frozen=True)
class RetrievalConfidence:
    top_score: float  # best similarity score, 0.0 if nothing retrieved
    score_gap: float  # top1 - top2, 0.0 if fewer than 2 chunks
    is_confident: bool
    reason: str  # human-readable; reused in logs and ticket drafts


def assess_confidence(
    scored_documents: list[tuple[Any, float]],
    min_top_score: float = DEFAULT_MIN_TOP_SCORE,
) -> RetrievalConfidence:
    """
    Judge whether retrieval found docs worth answering from.

    Args:
        scored_documents: (document, score) pairs, best first — the shape
            returned by a vector store's ``similarity_search_with_score``.
        min_top_score: Below this, the best match is treated as too weak to
            answer from.

    Returns:
        A RetrievalConfidence describing the verdict and why.
    """
    if not scored_documents:
        return RetrievalConfidence(
            top_score=0.0,
            score_gap=0.0,
            is_confident=False,
            reason="no chunks retrieved",
        )

    top_score = scored_documents[0][1]
    score_gap = top_score - scored_documents[1][1] if len(scored_documents) > 1 else 0.0

    if top_score < min_top_score:
        return RetrievalConfidence(
            top_score=top_score,
            score_gap=score_gap,
            is_confident=False,
            reason=f"best match is weak (top score {top_score:.3f} < {min_top_score})",
        )

    return RetrievalConfidence(
        top_score=top_score,
        score_gap=score_gap,
        is_confident=True,
        reason=f"top score {top_score:.3f} >= {min_top_score}",
    )

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

Scores are expected on the normalized [0, 1] relevance scale that
`retriever.retrieve_scored` returns (0 = dissimilar, 1 = most similar), which
is what lets one threshold hold across backends with different native metrics.
A relevance score is still not a probability — a usable threshold depends on
the embedding model AND the corpus — which is why `min_top_score` is a
caller-supplied parameter rather than a constant here, and why passing None
turns the check off for corpora where no threshold has been established yet.
"""

from dataclasses import dataclass
from typing import Any

from langsmith import traceable

#: On the normalized [0, 1] relevance scale. The pre-normalization default was
#: 0.35 against a raw Pinecone cosine score; cosine maps as (raw + 1) / 2, so
#: the equivalent bar is 0.675. Re-baseline against your own corpus.
DEFAULT_MIN_TOP_SCORE = 0.675


@dataclass(frozen=True)
class RetrievalConfidence:
    top_score: float  # best relevance score, 0.0 if nothing retrieved
    score_gap: float  # top1 - top2, 0.0 if fewer than 2 chunks
    is_confident: bool
    reason: str  # human-readable; reused in logs and ticket drafts


# @traceable(name="rag_core.assess_confidence")
def assess_confidence(
    scored_documents: list[tuple[Any, float]],
    min_top_score: float | None = DEFAULT_MIN_TOP_SCORE,
) -> RetrievalConfidence:
    """
    Judge whether retrieval found docs worth answering from.

    Args:
        scored_documents: (document, score) pairs, best first — the shape
            returned by :func:`rag_core.retriever.retrieve_scored`, on the
            normalized [0, 1] relevance scale.
        min_top_score: Below this, the best match is treated as too weak to
            answer from. Pass None to disable the check entirely — retrieval
            is then confident whenever it returned anything, which is the
            setting for a corpus with no established threshold yet.

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

    if min_top_score is None:
        return RetrievalConfidence(
            top_score=top_score,
            score_gap=score_gap,
            is_confident=True,
            reason="confidence check disabled (min_top_score is null)",
        )

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

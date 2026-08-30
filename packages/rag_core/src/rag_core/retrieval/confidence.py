"""Explainable retrieval-confidence heuristic.

Drives escalation in an agent: if retrieval looks weak, the agent should say
so (or offer a human hand-off) instead of bluffing an answer.

The heuristic (deliberately simple):
  1. No chunks retrieved                 -> not confident.
  2. Best cosine score < min_top_score   -> not confident ("best match is weak").
  3. Otherwise                           -> confident.

`score_gap` (top1 - top2) is computed and reported but does NOT affect the
decision — it is surfaced so it can be watched on real queries (a big gap
suggests one clearly-best doc; a flat top-k can mean the query matched
everything a little and nothing well).

Cosine similarity is not a probability: a usable threshold depends on the
embedding model AND the corpus, which is why `min_top_score` is per-project
configuration (`retrieval.min_top_score`) rather than a constant here.
"""

from dataclasses import dataclass

from langsmith import traceable

from rag_core.config import DEFAULT_MIN_TOP_SCORE
from rag_core.retrieval.retriever import RetrievedChunk


@dataclass(frozen=True)
class RetrievalConfidence:
    top_score: float  # best cosine score, 0.0 if nothing retrieved
    score_gap: float  # top1 - top2, 0.0 if fewer than 2 chunks
    is_confident: bool
    reason: str  # human-readable; reused in logs and ticket drafts


# Traced so the confidence verdict that drives escalation routing shows up as
# its own span. No-op unless LANGSMITH_TRACING is on.
@traceable
def assess_confidence(
    chunks: list[RetrievedChunk],
    min_top_score: float = DEFAULT_MIN_TOP_SCORE,
) -> RetrievalConfidence:
    """Judge whether retrieval found docs worth answering from."""
    if not chunks:
        return RetrievalConfidence(
            top_score=0.0,
            score_gap=0.0,
            is_confident=False,
            reason="no chunks retrieved",
        )

    top_score = chunks[0].score
    score_gap = chunks[0].score - chunks[1].score if len(chunks) > 1 else 0.0

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

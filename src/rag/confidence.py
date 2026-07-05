"""Task 2.3 — Explainable retrieval-confidence heuristic.

Drives escalation in Phase 3: if retrieval looks weak, the agent should offer
a Jira ticket instead of bluffing an answer.

The heuristic (deliberately simple):
  1. No chunks retrieved            -> not confident.
  2. Best cosine score < MIN_TOP_SCORE -> not confident ("best match is weak").
  3. Otherwise                      -> confident.

`score_gap` (top1 - top2) is computed and reported but does NOT affect the
decision yet — it's surfaced so we can watch it on real queries and decide in
Phase 5 evals whether it adds signal (a big gap suggests one clearly-best doc;
a flat top-k can mean the query matched everything a little and nothing well).

Cosine similarity is not a probability: usable thresholds depend on the
embedding model AND the corpus. MIN_TOP_SCORE below was eyeballed with the
sanity-check CLI (`python -m src.ingest.sanity_check`) on this corpus with
text-embedding-3-small; tune it in Phase 5.
"""

from dataclasses import dataclass

from langsmith import traceable

from src.rag.retriever import RetrievedChunk

# On-corpus questions scored ~0.4-0.6 top-1 in sanity checks; clearly
# off-corpus ones landed below ~0.3. 0.35 splits those with a little margin.
MIN_TOP_SCORE = 0.35


@dataclass(frozen=True)
class RetrievalConfidence:
    top_score: float  # best cosine score, 0.0 if nothing retrieved
    score_gap: float  # top1 - top2, 0.0 if fewer than 2 chunks
    is_confident: bool
    reason: str  # human-readable; reused later in logs and the Jira draft


# Task 5.2 — traced so the confidence verdict that drives escalation routing
# shows up as its own span. No-op unless LANGSMITH_TRACING is on.
@traceable
def assess_confidence(chunks: list[RetrievedChunk]) -> RetrievalConfidence:
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

    if top_score < MIN_TOP_SCORE:
        return RetrievalConfidence(
            top_score=top_score,
            score_gap=score_gap,
            is_confident=False,
            reason=f"best match is weak (top score {top_score:.3f} < {MIN_TOP_SCORE})",
        )

    return RetrievalConfidence(
        top_score=top_score,
        score_gap=score_gap,
        is_confident=True,
        reason=f"top score {top_score:.3f} >= {MIN_TOP_SCORE}",
    )

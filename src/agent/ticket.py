"""Task 3.5 — Build a Jira ticket DRAFT from the agent state.

Pure and deterministic: no LLM, no network, fully unit-testable. The draft
is assembled from facts the graph already collected — the question, the last
answer, the confidence verdict, and the docs-checked list from chunk metadata
(the reason task 1.3 attached heading/url to every chunk). Phase 4 turns this
into a real Jira payload; an LLM-polished description is a possible later
enhancement.

Note: this module takes the state as a plain mapping instead of importing
`AgentState`, because state.py imports `TicketDraft` from here — typing it
as AgentState would create a circular import.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from src.rag.retriever import RetrievedChunk

# Generic but actionable defaults for a support engineer picking up the
# ticket. Static on purpose — deterministic drafts are easy to test and
# can never hallucinate steps.
DEFAULT_NEXT_STEPS = [
    "Review the docs listed under 'Docs checked' to confirm coverage gaps.",
    "Ask the reporter for exact error messages, logs, and the region/account.",
    "Reproduce the issue in a sandbox pipeline if possible.",
]


@dataclass(frozen=True)
class TicketDraft:
    summary: str
    description: str
    docs_checked: list[str] = field(default_factory=list)
    suggested_next_steps: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Human-readable form for printing/logging the draft."""
        docs = "\n".join(f"  - {d}" for d in self.docs_checked) or "  (none)"
        steps = "\n".join(f"  {i}. {s}" for i, s in enumerate(self.suggested_next_steps, 1))
        return (
            f"Summary: {self.summary}\n"
            f"Description:\n{self.description}\n"
            f"Docs checked:\n{docs}\n"
            f"Suggested next steps:\n{steps}"
        )


def _docs_checked(chunks: list[RetrievedChunk]) -> list[str]:
    """Deduped 'heading — url' lines, preserving retrieval-rank order."""
    seen: dict[str, None] = {}  # dict as an ordered set
    for chunk in chunks:
        line = f"{chunk.heading} — {chunk.url}"
        seen.setdefault(line, None)
    return list(seen)


def build_ticket_draft(state: Mapping) -> TicketDraft:
    """Assemble the draft from whatever the graph learned before escalating."""
    question = state["question"]
    answer = state.get("answer")
    confidence = state.get("confidence")
    attempts = state.get("attempts", 0)

    description_parts = [
        f"User question: {question}",
        f"Retrieval attempts: {attempts}",
        (
            f"Retrieval confidence: {confidence.reason}"
            if confidence
            else "Retrieval confidence: not assessed"
        ),
        (
            f"Last automated answer (did not resolve the issue):\n{answer}"
            if answer
            else "No automated answer was produced."
        ),
    ]

    return TicketDraft(
        summary=f"Unresolved AWS CI/CD issue: {question}",
        description="\n\n".join(description_parts),
        docs_checked=_docs_checked(state.get("chunks", [])),
        suggested_next_steps=list(DEFAULT_NEXT_STEPS),
    )

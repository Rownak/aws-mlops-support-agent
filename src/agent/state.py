"""Task 3.1 — The graph state schema.

The state is the ONLY thing LangGraph passes between nodes, and the only
thing the checkpointer persists when the graph pauses at an interrupt.
Nodes never mutate it: each node returns a PARTIAL dict (just the keys it
changed) and LangGraph merges that into the checkpointed state before the
next node runs. That merge-not-mutate contract is what makes pausing,
resuming, and replaying a run possible.
"""

from typing import Literal, TypedDict

from src.agent.ticket import TicketDraft
from src.rag.confidence import RetrievalConfidence
from src.rag.retriever import RetrievedChunk


class AgentState(TypedDict):
    # The user's question. Written once at graph start, never changed.
    question: str
    # Docs from the LATEST retrieve attempt (each retry overwrites, not
    # appends — the answer should be grounded in one coherent set).
    chunks: list[RetrievedChunk]
    # Latest generated answer; None until the answer node has run.
    answer: str | None
    # Completed retrieve→answer cycles. The graph has a loop (retry goes
    # back to retrieve), and graphs are stateless between steps — so the
    # loop counter must live IN the state, not in a Python variable.
    attempts: int
    # Heuristic verdict on the latest retrieval; drives the
    # confident-vs-escalate conditional edge.
    confidence: RetrievalConfidence | None
    # True only when the user explicitly confirmed the answer solved it.
    resolved: bool
    # What the user chose at the confirm_resolution interrupt
    # ("user satisfaction" in the task description). None = not asked yet.
    user_action: Literal["resolved", "retry", "ticket"] | None
    # Populated only by the escalate node; Phase 4 sends this to Jira.
    ticket_draft: TicketDraft | None
    # What the Jira wrapper returned: the dry-run echo (payload logged, not
    # sent), or the real Jira response with the created issue key. None if
    # escalation didn't happen, was cancelled, or Jira isn't configured.
    ticket_result: dict | None


def initial_state(question: str) -> AgentState:
    """Fresh state for a new question — one place that defines the defaults."""
    return AgentState(
        question=question,
        chunks=[],
        answer=None,
        attempts=0,
        confidence=None,
        resolved=False,
        user_action=None,
        ticket_draft=None,
        ticket_result=None,
    )

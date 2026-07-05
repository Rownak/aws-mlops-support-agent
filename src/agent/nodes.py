"""Tasks 3.2 / 3.3 / 3.5 — The graph's node functions.

A LangGraph node is just a function `state -> partial state update`. The
nodes here wrap the Phase 2 RAG functions; the graph wiring lives in
graph.py.

`build_nodes` is a factory: tests inject a fake `retriever`/`answerer` so
the whole graph runs without Pinecone or OpenAI; production callers pass
nothing and get the real ones bound to `cfg`.

How the interrupt (task 3.3) works
----------------------------------
`interrupt(payload)` inside a node does NOT block waiting for input — there
is no one to type anything inside a server process. Instead it:
  1. raises internally, stopping the graph run;
  2. the checkpointer saves the full state under the caller's `thread_id`;
  3. the caller's `invoke()` returns with an `__interrupt__` entry carrying
     `payload` (whatever we want the human to see).
Later — seconds or days — the caller resumes the SAME thread with
`invoke(Command(resume=value), config={thread_id: ...})`. LangGraph reloads
the checkpoint, re-runs this node from its start, and this time
`interrupt()` RETURNS `value` instead of pausing. That "re-runs from the
node's start" detail is why interrupt nodes should do nothing expensive
before the interrupt call.
"""

from collections.abc import Callable

from langgraph.types import interrupt

from src.agent.ticket import build_ticket_draft
from src.config import Config
from src.rag.confidence import assess_confidence

# The choices offered at the confirm_resolution interrupt. app.py shows
# them to the human; tests pass them via Command(resume=...).
USER_ACTIONS = ("resolved", "retry", "ticket")


def build_nodes(
    cfg: Config,
    retriever: Callable | None = None,
    answerer: Callable | None = None,
) -> dict[str, Callable]:
    """Return {node_name: node_fn}, with real RAG functions unless injected."""
    if retriever is None:
        # Imported lazily: building the real retriever opens a Pinecone
        # connection, which tests (that always inject) must never do.
        from src.rag.retriever import make_retriever

        retriever = make_retriever(cfg)
    if answerer is None:
        from src.rag.answer import generate_answer

        def answerer(question, chunks):  # noqa: F811 - deliberate rebind
            return generate_answer(question, chunks, cfg)

    def retrieve(state) -> dict:
        """Task 3.2 — fetch docs, judge them, count the attempt."""
        # Retries widen the net (k grows) instead of re-running the exact
        # same top-4 search, so a second attempt can actually differ.
        k = 4 + 2 * state["attempts"]
        chunks = retriever(state["question"], k=k)
        return {
            "chunks": chunks,
            "confidence": assess_confidence(chunks),
            "attempts": state["attempts"] + 1,
        }

    def answer(state) -> dict:
        """Task 3.2 — generate the cited answer from the latest chunks."""
        return {"answer": answerer(state["question"], state["chunks"])}

    def confirm_resolution(state) -> dict:
        """Task 3.3 — pause and ask the human whether we're done."""
        action = interrupt(
            {
                "answer": state["answer"],
                "question": "Did this answer resolve your issue?",
                "options": list(USER_ACTIONS),
            }
        )
        if action not in USER_ACTIONS:
            # A bad resume value must not silently count as "resolved".
            raise ValueError(f"Unexpected resume value {action!r}; expected one of {USER_ACTIONS}")
        return {"user_action": action, "resolved": action == "resolved"}

    def escalate(state) -> dict:
        """Task 3.5 — draft (don't send!) a Jira ticket and show it."""
        draft = build_ticket_draft(state)
        print("\n=== Jira ticket draft (NOT sent — Phase 4 wires the API) ===")
        print(draft.render())
        return {"ticket_draft": draft}

    return {
        "retrieve": retrieve,
        "answer": answer,
        "confirm_resolution": confirm_resolution,
        "escalate": escalate,
    }

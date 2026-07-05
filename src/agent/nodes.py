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

from src.agent.jira_tool import create_issue
from src.agent.ticket import build_ticket_draft
from src.config import Config
from src.rag.confidence import assess_confidence

# The choices offered at the confirm_resolution interrupt. app.py shows
# them to the human; tests pass them via Command(resume=...).
USER_ACTIONS = ("resolved", "retry", "ticket")

# The choices offered at the confirm_ticket interrupt (task 4.2): the
# safety confirmation before a REAL (non-dry-run) Jira ticket is created.
TICKET_ACTIONS = ("create", "cancel")


def build_nodes(
    cfg: Config,
    retriever: Callable | None = None,
    answerer: Callable | None = None,
    jira_creator: Callable | None = None,
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

    if jira_creator is None:
        # The real wrapper from task 4.1. Safe as a default even in tests
        # that forget to inject: with dry_run=True it never touches the
        # network, and with unset Jira vars it raises RuntimeError, which
        # escalate() catches below.
        def jira_creator(draft):  # noqa: F811 - deliberate rebind
            return create_issue(draft, cfg)

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
                # "type" lets app.py tell this pause apart from the
                # confirm_ticket pause in escalate — both arrive through
                # the same __interrupt__ channel.
                "type": "confirm_resolution",
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
        """Tasks 3.5 + 4.2 — draft a Jira ticket and hand it to the wrapper.

        Dry-run (the default): no pause — create_issue just logs the payload.
        Live mode: a second interrupt asks the human to confirm BEFORE any
        real ticket is created (CLAUDE.md: never auto-create without a
        confirmation step). Two escalation triggers — low confidence and
        attempts exhausted — reach here without the user ever asking for a
        ticket, so the DRY_RUN flag alone isn't confirmation enough.
        """
        # Building the draft is pure and cheap, so it's safe that this line
        # re-runs when the node restarts after the interrupt resumes.
        draft = build_ticket_draft(state)

        if not cfg.dry_run:
            # Nothing may be printed before this call: on resume the node
            # re-runs from the top, and any earlier print would show twice.
            # The rendered draft rides in the payload; app.py displays it.
            decision = interrupt(
                {
                    "type": "confirm_ticket",
                    "draft": draft.render(),
                    "question": "Create this ticket in Jira for real?",
                    "options": list(TICKET_ACTIONS),
                }
            )
            if decision not in TICKET_ACTIONS:
                # A bad resume value must not silently create a real ticket.
                raise ValueError(
                    f"Unexpected resume value {decision!r}; expected one of {TICKET_ACTIONS}"
                )
            if decision == "cancel":
                print("\n=== Jira ticket draft (cancelled — not sent) ===")
                print(draft.render())
                return {"ticket_draft": draft, "ticket_result": None}

        print("\n=== Escalating to Jira ===")
        try:
            # Dry-run: create_issue prints the payload and returns without
            # any network call. Live + confirmed: creates the real issue.
            result = jira_creator(draft)
        except RuntimeError as exc:
            # Jira env vars not set — the demo must still work without
            # them, so show the draft instead of crashing the graph.
            print(f"(Jira not configured: {exc} Draft shown below, not sent.)")
            print(draft.render())
            return {"ticket_draft": draft, "ticket_result": None}
        return {"ticket_draft": draft, "ticket_result": result}

    return {
        "retrieve": retrieve,
        "answer": answer,
        "confirm_resolution": confirm_resolution,
        "escalate": escalate,
    }

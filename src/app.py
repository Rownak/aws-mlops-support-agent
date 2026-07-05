"""Interactive agent runner — the human side of the interrupt loop.

Usage:  uv run python -m src.app

Flow per question:
  1. invoke the graph with a fresh state and a unique thread_id;
  2. if the run PAUSES (result carries "__interrupt__"), the payload's
     "type" says which pause it is: confirm_resolution (show the answer,
     ask resolved / ticket / ask again) or confirm_ticket (live mode only:
     show the draft, ask whether to really create the Jira ticket);
  3. resume the SAME thread with Command(resume=<choice>) and repeat until
     the graph reaches END.

The thread_id is how the checkpointer knows which paused run to reload —
same id = same conversation. A new question gets a new id (fresh state).
"""

import uuid

from langgraph.types import Command

from src.agent.graph import build_graph
from src.agent.state import initial_state
from src.config import load_config
from src.observability import log_event, setup_json_logging

# Map single keypresses to the resume values confirm_resolution expects.
CHOICES = {"r": "resolved", "t": "ticket", "a": "retry"}


def _ask_resolution_choice() -> str:
    while True:
        raw = input("\nDid this resolve your issue? [r]esolved / [t]icket / [a]sk again: ")
        choice = raw.strip().lower()
        if choice in CHOICES:
            return CHOICES[choice]
        print(f"Please type one of: {', '.join(CHOICES)}")


def _ask_ticket_confirmation() -> str:
    while True:
        raw = input("\nCreate this ticket in Jira FOR REAL? [y]es / [n]o: ")
        choice = raw.strip().lower()
        if choice in ("y", "yes"):
            return "create"
        if choice in ("n", "no"):
            return "cancel"
        print("Please type y or n")


def _handle_interrupt(result) -> str:
    """Show the pause's payload to the human, return the resume value."""
    payload = result["__interrupt__"][0].value
    if payload["type"] == "confirm_ticket":
        print("\n=== Jira ticket draft ===")
        print(payload["draft"])
        return _ask_ticket_confirmation()
    # confirm_resolution: the answer travels in the payload (state["answer"]
    # would also work, but the payload is the interrupt's contract).
    print(f"\n=== Answer (attempt {result['attempts']}) ===")
    print(payload["answer"])
    return _ask_resolution_choice()


def run_question(graph, question: str, cfg) -> None:
    thread_id = str(uuid.uuid4())
    thread = {"configurable": {"thread_id": thread_id}}
    # thread_id correlates this line with the node events of the same run
    # (and with the LangSmith trace, which records the same config).
    log_event("question_received", thread_id=thread_id, question=question)
    result = graph.invoke(initial_state(question), thread)

    # Loop while the graph keeps pausing (resolution and/or ticket confirm).
    while "__interrupt__" in result:
        result = graph.invoke(Command(resume=_handle_interrupt(result)), thread)

    if result["resolved"]:
        print("\nGreat — marked as resolved.")
    ticket_result = result.get("ticket_result")
    if ticket_result and "key" in ticket_result:
        # A real ticket was created — show where to find it.
        print(f"\nTicket created: {cfg.jira_base_url.rstrip('/')}/browse/{ticket_result['key']}")


def main() -> None:
    cfg = load_config()  # also loads .env — incl. LANGSMITH_* vars if set
    setup_json_logging()
    graph = build_graph(cfg)
    print("AWS CI/CD support agent — CodeBuild & CodePipeline docs.")
    print("Empty question quits.")
    while True:
        question = input("\nYour question: ").strip()
        if not question:
            break
        run_question(graph, question, cfg)


if __name__ == "__main__":
    main()

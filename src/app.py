"""Interactive agent runner — the human side of the interrupt loop.

Usage:  uv run python -m src.app

Flow per question:
  1. invoke the graph with a fresh state and a unique thread_id;
  2. if the run PAUSES (result carries "__interrupt__"), show the answer
     and ask resolved / ticket / ask again;
  3. resume the SAME thread with Command(resume=<choice>) and repeat until
     the graph reaches END (resolved, or a printed ticket draft).

The thread_id is how the checkpointer knows which paused run to reload —
same id = same conversation. A new question gets a new id (fresh state).
"""

import uuid

from langgraph.types import Command

from src.agent.graph import build_graph
from src.agent.state import initial_state
from src.config import load_config

# Map single keypresses to the resume values confirm_resolution expects.
CHOICES = {"r": "resolved", "t": "ticket", "a": "retry"}


def _ask_user_choice() -> str:
    while True:
        raw = input("\nDid this resolve your issue? [r]esolved / [t]icket / [a]sk again: ")
        choice = raw.strip().lower()
        if choice in CHOICES:
            return CHOICES[choice]
        print(f"Please type one of: {', '.join(CHOICES)}")


def run_question(graph, question: str) -> None:
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = graph.invoke(initial_state(question), thread)

    # Loop while the graph keeps pausing at confirm_resolution.
    while "__interrupt__" in result:
        print(f"\n=== Answer (attempt {result['attempts']}) ===")
        print(result["answer"])
        result = graph.invoke(Command(resume=_ask_user_choice()), thread)

    if result["resolved"]:
        print("\nGreat — marked as resolved.")
    # The escalate node already printed the ticket draft; nothing to add.


def main() -> None:
    graph = build_graph(load_config())
    print("AWS CI/CD support agent — CodeBuild & CodePipeline docs.")
    print("Empty question quits.")
    while True:
        question = input("\nYour question: ").strip()
        if not question:
            break
        run_question(graph, question)


if __name__ == "__main__":
    main()

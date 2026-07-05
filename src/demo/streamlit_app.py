"""Task 6.1 — Streamlit demo UI: a tiny chat page over the agent graph.

Run:  uv run streamlit run src/demo/streamlit_app.py

Streamlit reruns this whole script top-to-bottom on every click or message,
so src/app.py's blocking `while "__interrupt__"` loop can't be reused here.
Instead the graph's pause lives in st.session_state: each user interaction
triggers exactly one graph.invoke, and the script re-renders from whatever
state that left behind.

Demo mode: `demo_config()` forces dry_run=True no matter what .env says, so
no real Jira ticket can ever be created from the public demo. That also
means the escalate node's confirm_ticket interrupt (live mode only) can
never fire — the UI only handles the confirm_resolution pause.
"""

import dataclasses
import sys
import uuid
from pathlib import Path

# `streamlit run` puts the SCRIPT's dir (src/demo) on sys.path, not the
# project root — insert the root so the `from src...` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
from langgraph.types import Command

from src.agent.graph import build_graph
from src.agent.state import initial_state
from src.config import Config, load_config
from src.observability import log_event, setup_json_logging


def demo_config() -> Config:
    """Public demo: Jira is ALWAYS dry-run, regardless of what .env says."""
    return dataclasses.replace(load_config(), dry_run=True)


@st.cache_resource
def get_graph():
    # Built once per server process, not per rerun: the InMemorySaver
    # checkpointer inside the compiled graph holds every paused thread,
    # so rebuilding it would orphan conversations waiting on a button.
    setup_json_logging()
    return build_graph(demo_config())


def render_outcome(result) -> str:
    """Message for a finished run (graph reached END) — reads state only."""
    if result["resolved"]:
        return "Great — marked as resolved."
    draft = result["ticket_draft"]
    ticket_result = result["ticket_result"]
    if ticket_result and ticket_result.get("dry_run"):
        header = "**Demo mode** — ticket drafted but NOT sent to Jira (dry-run)."
    else:
        # Jira env vars unset: escalate absorbed the RuntimeError and
        # returned ticket_result=None with the draft (nodes.py).
        header = "Jira isn't configured — ticket draft shown, not sent."
    return f"{header}\n```text\n{draft.render()}\n```"


def handle_result(result, thread_id: str) -> None:
    """Route one invoke() result into session state (paused vs finished)."""
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        if payload["type"] != "confirm_resolution":
            # demo_config forces dry_run=True, so escalate's confirm_ticket
            # interrupt is unreachable here. Fail loudly if that changes.
            raise RuntimeError(f"Unexpected interrupt type: {payload['type']!r}")
        st.session_state.history.append({"role": "assistant", "content": payload["answer"]})
        st.session_state.pending = {"thread_id": thread_id, "question": payload["question"]}
    else:
        st.session_state.pending = None
        st.session_state.history.append({"role": "assistant", "content": render_outcome(result)})


def main() -> None:
    st.title("AWS CI/CD support agent")
    st.caption(
        "Answers from the CodeBuild & CodePipeline docs. "
        "Demo mode: Jira escalation is dry-run only — no real tickets."
    )
    graph = get_graph()
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("pending", None)

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pending = st.session_state.pending
    if pending:
        st.write(pending["question"])  # "Did this answer resolve your issue?"
        # Button labels map onto the resume values confirm_resolution accepts.
        cols = st.columns(3)
        actions = [("Resolved", "resolved"), ("Ask again", "retry"), ("Open a ticket", "ticket")]
        for col, (label, action) in zip(cols, actions, strict=True):
            if col.button(label):
                thread = {"configurable": {"thread_id": pending["thread_id"]}}
                with st.spinner("Working..."):
                    result = graph.invoke(Command(resume=action), thread)
                handle_result(result, pending["thread_id"])
                st.rerun()

    question = st.chat_input("Ask about CodeBuild / CodePipeline", disabled=pending is not None)
    if question:
        # Fresh thread per question, same as the CLI: the graph is
        # single-turn, so each question is its own checkpointed run.
        thread_id = str(uuid.uuid4())
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):  # show it now; the rerun re-renders it
            st.markdown(question)
        log_event("question_received", thread_id=thread_id, question=question, ui="streamlit")
        thread = {"configurable": {"thread_id": thread_id}}
        with st.spinner("Thinking..."):
            result = graph.invoke(initial_state(question), thread)
        handle_result(result, thread_id)
        st.rerun()


if __name__ == "__main__":  # `streamlit run` executes with __name__ == "__main__",
    main()  # so the module stays import-safe for tests

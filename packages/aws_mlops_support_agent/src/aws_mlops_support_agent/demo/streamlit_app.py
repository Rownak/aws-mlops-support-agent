"""Streamlit demo UI: a tiny chat page over the agent graph.

Run:  uv run aws-agent-demo

Streamlit reruns this whole script top-to-bottom on every click or message,
so app.py's blocking `while "__interrupt__"` loop can't be reused here.
Instead the graph's pause lives in st.session_state: each user interaction
triggers exactly one graph.invoke, and the script re-renders from whatever
state that left behind.

Demo mode: `demo_config()` forces dry_run=True no matter what .env says, so
no real Jira ticket can ever be created from the public demo. That also
means the escalate node's confirm_ticket interrupt (live mode only) can
never fire — the UI only handles the confirm_resolution pause.
"""

import uuid

import streamlit as st
from langgraph.types import Command
from rag_core.observability import log_event, setup_json_logging

from aws_mlops_support_agent.agent.graph import build_graph
from aws_mlops_support_agent.agent.state import initial_state
from aws_mlops_support_agent.settings import AgentConfig, force_dry_run, load_settings

# Every LLM-spending action a visitor can take is metered against this.
# Deliberately small: this is a shared demo key, not a product.
#
# NB this is a courtesy limit, not a security control — st.session_state is
# tied to the WebSocket connection, so refreshing the page starts a new
# session with a fresh count. The real protection against key exhaustion is
# the hard monthly spend cap on the OpenAI project key (see
# deploy/streamlit_cloud_setup.md), which is what actually cannot be bypassed.
MAX_RUNS_PER_SESSION = 5


def demo_config() -> AgentConfig:
    """Public demo: Jira is ALWAYS dry-run, regardless of what .env says."""
    return force_dry_run(load_settings())


def runs_left() -> int:
    """How many LLM-backed runs this session may still make."""
    return MAX_RUNS_PER_SESSION - st.session_state.get("runs_used", 0)


def record_run() -> None:
    """Count one LLM-backed graph run against this session's quota."""
    st.session_state.runs_used = st.session_state.get("runs_used", 0) + 1


@st.cache_resource
def get_graph():
    # Built once per server process, not per rerun: the InMemorySaver
    # checkpointer inside the compiled graph holds every paused thread,
    # so rebuilding it would orphan conversations waiting on a button.
    setup_json_logging()
    return build_graph(demo_config())


def render_ingest_summary(stats) -> None:
    """Render one IngestStats (rag_core.ingestion.IngestStats) in the sidebar."""
    st.success(
        f"{stats['processed']}/{stats['total']} documents processed "
        f"({stats['skipped']} skipped, {stats['failed']} failed, "
        f"{stats['replaced']} replaced, {stats['chunks_created']} chunks)"
    )
    if stats["errors"]:
        with st.expander(f"{len(stats['errors'])} error(s)"):
            for err in stats["errors"]:
                st.text(f"{err['file']}: {err['error']}")


def render_ingest_control() -> None:
    """Sidebar button that runs RagCore.sync() against this project's config.yml.

    Disabled in the hosted demo: that image is built without rag_core's
    `ingest` extra (no markitdown/onnxruntime/etc — see rag_core pyproject),
    so RagCore.sync() would fail with an ImportError if this ever ran there.
    The demo queries the pre-built Pinecone index and never needs to ingest.
    """
    st.sidebar.subheader("Corpus")
    source_ids = [spec.options["id"] for spec in demo_config().rag.sources]
    st.sidebar.caption("Docs covered: " + ", ".join(source_ids))
    st.sidebar.caption("Ingestion disabled in this demo — querying a pre-built index.")
    if st.sidebar.button("Ingest / refresh docs", disabled=True):
        # Imported lazily, same reasoning as nodes.py: building RagCore opens
        # a Pinecone connection, which should not happen on every page render.
        from rag_core import RagCore

        # Registers awsdocs_git with rag_core's REGISTRY (see ingest.py) —
        # without this, sync() fails with "Unknown source type: awsdocs_git".
        import aws_mlops_support_agent.sources  # noqa: F401
        from aws_mlops_support_agent.settings import CONFIG_PATH

        with st.sidebar:
            with st.spinner("Ingesting (clone, chunk, embed)..."):
                stats = RagCore(str(CONFIG_PATH)).sync()
            render_ingest_summary(stats)


def render_answer(answer: str, citations: list[str]) -> str:
    """Chat message for a paused (confirm_resolution) run: answer + sources."""
    if not citations:
        return answer
    sources = "\n".join(f"- {line}" for line in citations)
    return f"{answer}\n\n**Sources:**\n{sources}"


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
        message = render_answer(payload["answer"], payload["citations"])
        st.session_state.history.append({"role": "assistant", "content": message})
        st.session_state.pending = {"thread_id": thread_id, "question": payload["question"]}
    else:
        st.session_state.pending = None
        st.session_state.history.append({"role": "assistant", "content": render_outcome(result)})


def main() -> None:
    st.title("AWS CI/CD support agent")
    st.caption(
        "Answers from the CodeBuild, CodePipeline, CodeDeploy & AmazonECR docs. "
        "Demo mode: Jira escalation is dry-run only — no real tickets."
    )
    graph = get_graph()
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("pending", None)
    st.session_state.setdefault("runs_used", 0)
    render_ingest_control()

    st.sidebar.subheader("Demo quota")
    st.sidebar.caption(f"{runs_left()} of {MAX_RUNS_PER_SESSION} questions left this session.")

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
            # "Ask again" re-runs the graph and costs another LLM call, so it
            # spends quota too — otherwise retry would be an unmetered loop.
            # "Resolved"/"Open a ticket" just finish the run (ticket drafting
            # is local, and Jira is dry-run), so they stay free.
            costs_quota = action == "retry"
            if col.button(label, disabled=costs_quota and runs_left() <= 0):
                if costs_quota:
                    record_run()
                thread = {"configurable": {"thread_id": pending["thread_id"]}}
                with st.spinner("Working..."):
                    result = graph.invoke(Command(resume=action), thread)
                handle_result(result, pending["thread_id"])
                st.rerun()

    if runs_left() <= 0 and not pending:
        st.info(
            f"Demo limit reached ({MAX_RUNS_PER_SESSION} questions per session). "
            "Refresh the page to start a new session."
        )

    question = st.chat_input(
        "Ask about CodeBuild / CodePipeline",
        disabled=pending is not None or runs_left() <= 0,
    )
    if question:
        record_run()
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

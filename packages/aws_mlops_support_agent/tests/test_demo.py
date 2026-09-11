"""Tests for the Streamlit demo module.

Only the pure parts are tested: the demo-mode dry-run guarantee and the
outcome-message rendering. The UI itself runs under `if __name__ ==
"__main__"`, so importing the module here is side-effect free (Streamlit
calls in bare mode are no-ops).
"""

from rag_core.ingestion import empty_stats

from aws_mlops_support_agent.agent.ticket import TicketDraft
from aws_mlops_support_agent.demo.streamlit_app import (
    demo_config,
    render_answer,
    render_ingest_control,
    render_ingest_summary,
    render_outcome,
)

DRAFT = TicketDraft(
    summary="Unresolved AWS CI/CD issue: pipeline stuck",
    description="User question: pipeline stuck",
    docs_checked=["Concepts — https://example.com"],
    suggested_next_steps=["Ask for logs."],
)


def test_demo_config_forces_dry_run(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("PINECONE_API_KEY", "fake")
    monkeypatch.setenv("DRY_RUN", "false")  # explicit live mode in the env...
    assert demo_config().dry_run is True  # ...still forced back to dry-run


def test_render_outcome_resolved():
    result = {"resolved": True}
    assert render_outcome(result) == "Great — marked as resolved."


def test_render_outcome_dry_run_ticket():
    result = {
        "resolved": False,
        "ticket_draft": DRAFT,
        "ticket_result": {"dry_run": True, "url": "https://x/rest/api/3/issue", "payload": {}},
    }
    message = render_outcome(result)
    assert "NOT sent to Jira" in message
    assert DRAFT.render() in message


def test_render_outcome_jira_unconfigured():
    result = {"resolved": False, "ticket_draft": DRAFT, "ticket_result": None}
    message = render_outcome(result)
    assert "isn't configured" in message
    assert DRAFT.render() in message


def test_render_ingest_summary_runs_on_a_clean_result():
    # Streamlit calls are no-ops in bare mode; this just checks the function
    # doesn't raise on a real IngestStats shape (rag_core.ingestion.IngestStats).
    stats = empty_stats(total=3)
    stats["processed"] = 3
    stats["chunks_created"] = 12
    render_ingest_summary(stats)


def test_render_ingest_summary_runs_with_errors():
    stats = empty_stats(total=2)
    stats["processed"] = 1
    stats["failed"] = 1
    stats["errors"] = [{"file": "a.md", "error": "boom"}]
    render_ingest_summary(stats)


def test_render_answer_without_citations_is_just_the_text():
    assert render_answer("Plain answer.", []) == "Plain answer."


def test_render_answer_appends_sources_list():
    message = render_answer(
        "CodeDeploy automates deployments [1].",
        ["[1] codedeploy — https://docs.aws/codedeploy.html"],
    )
    assert "CodeDeploy automates deployments [1]." in message
    assert "**Sources:**" in message
    assert "[1] codedeploy — https://docs.aws/codedeploy.html" in message


def test_render_ingest_control_lists_configured_source_ids(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("PINECONE_API_KEY", "fake")
    # Streamlit calls are no-ops in bare mode; this just checks the sidebar
    # source list is built from config.yml's real (uncommented) sources
    # without raising — not a fixed set, since which sources are enabled
    # changes as more guides get ingested locally — the whole point being
    # that this list can't silently drift from config.yml the way the demo's
    # hardcoded caption text used to.
    source_ids = {spec.options["id"] for spec in demo_config().rag.sources}
    assert source_ids  # at least one source is enabled
    render_ingest_control()

"""Task 6.1 — Tests for the Streamlit demo module.

Only the pure parts are tested: the demo-mode dry-run guarantee and the
outcome-message rendering. The UI itself runs under `if __name__ ==
"__main__"`, so importing the module here is side-effect free (Streamlit
calls in bare mode are no-ops).
"""

from src.agent.ticket import TicketDraft
from src.demo.streamlit_app import demo_config, render_outcome

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

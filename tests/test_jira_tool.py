"""Tests for task 4.1 — Jira REST wrapper. Fully offline: requests.post is
monkeypatched, never actually called over the network.
"""

import pytest

from src.agent import jira_tool
from src.agent.ticket import TicketDraft
from src.config import Config


def _cfg(**overrides) -> Config:
    fields = {
        "openai_api_key": "unused",
        "openai_chat_model": "unused",
        "openai_embedding_model": "unused",
        "pinecone_api_key": "unused",
        "pinecone_index_name": "unused",
        "aws_region": "unused",
        "jira_base_url": "https://example.atlassian.net",
        "jira_email": "dev@example.com",
        "jira_api_token": "token-123",
        "jira_project_key": "SUP",
        "dry_run": True,
    }
    fields.update(overrides)
    return Config(**fields)


def _draft() -> TicketDraft:
    return TicketDraft(
        summary="Unresolved AWS CI/CD issue: build fails",
        description="User question: why does my build fail?",
        docs_checked=["Buildspec — https://docs.aws/b.html"],
        suggested_next_steps=["Reproduce in a sandbox pipeline."],
    )


def test_missing_jira_config_raises_with_names():
    cfg = _cfg(jira_base_url=None, jira_project_key=None)
    with pytest.raises(RuntimeError, match="jira_base_url, jira_project_key"):
        jira_tool.create_issue(_draft(), cfg)


def test_dry_run_never_calls_requests(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("requests.post must not be called in dry-run")

    monkeypatch.setattr(jira_tool.requests, "post", _boom)

    result = jira_tool.create_issue(_draft(), _cfg(dry_run=True))

    assert result["dry_run"] is True
    assert result["payload"]["fields"]["project"]["key"] == "SUP"
    assert result["payload"]["fields"]["summary"] == _draft().summary


def test_live_call_posts_expected_request(monkeypatch):
    calls = []

    class FakeResponse:
        ok = True

        def json(self):
            return {"key": "SUP-42"}

    def fake_post(url, json, auth, timeout):
        calls.append({"url": url, "json": json, "auth": auth, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr(jira_tool.requests, "post", fake_post)

    result = jira_tool.create_issue(_draft(), _cfg(dry_run=False))

    assert result == {"key": "SUP-42"}
    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == "https://example.atlassian.net/rest/api/3/issue"
    assert call["auth"] == ("dev@example.com", "token-123")
    assert call["json"]["fields"]["project"]["key"] == "SUP"
    assert call["json"]["fields"]["issuetype"]["name"] == "Task"


def test_live_call_raises_on_http_error_with_body(monkeypatch):
    class FailingResponse:
        ok = False
        status_code = 400
        text = '{"errors":{"issuetype":"issue type is required"}}'

    monkeypatch.setattr(jira_tool.requests, "post", lambda *a, **k: FailingResponse())

    # The Jira error body must surface in the exception — a bare
    # "400 Bad Request" is undebuggable.
    with pytest.raises(RuntimeError, match="issue type is required"):
        jira_tool.create_issue(_draft(), _cfg(dry_run=False))


def test_trailing_slash_in_base_url_tolerated(monkeypatch):
    urls = []

    class FakeResponse:
        ok = True

        def json(self):
            return {"key": "SUP-1"}

    def fake_post(url, **kwargs):
        urls.append(url)
        return FakeResponse()

    monkeypatch.setattr(jira_tool.requests, "post", fake_post)

    cfg = _cfg(jira_base_url="https://example.atlassian.net/", dry_run=False)
    jira_tool.create_issue(_draft(), cfg)
    assert urls == ["https://example.atlassian.net/rest/api/3/issue"]

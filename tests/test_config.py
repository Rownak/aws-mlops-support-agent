"""Tests for src.config — run with: uv run pytest"""

import pytest

from src import config


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    """Keep tests independent of the developer's real .env / environment.

    Disables .env loading and clears every var the config reads, then sets
    the two required ones to dummy values. Individual tests override as needed.
    """
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_CHAT_MODEL",
        "OPENAI_EMBEDDING_MODEL",
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "AWS_REGION",
        "JIRA_BASE_URL",
        "JIRA_EMAIL",
        "JIRA_API_TOKEN",
        "JIRA_PROJECT_KEY",
        "DRY_RUN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")


def test_missing_required_var_fails_with_its_name(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        config.load_config()


def test_all_missing_vars_reported_at_once(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("PINECONE_API_KEY", "")  # empty counts as missing
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY, PINECONE_API_KEY"):
        config.load_config()


def test_defaults_applied_when_unset():
    cfg = config.load_config()
    assert cfg.dry_run is True  # the safety default
    assert cfg.openai_chat_model == "gpt-4o-mini"
    assert cfg.openai_embedding_model == "text-embedding-3-small"
    assert cfg.jira_base_url is None  # Jira is optional until Phase 4


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("false", False),
        ("FALSE", False),
        ("0", False),
        ("true", True),
        ("flase", True),  # typo must fail safe (stay in dry-run)
        ("", True),
    ],
)
def test_dry_run_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("DRY_RUN", raw)
    assert config.load_config().dry_run is expected

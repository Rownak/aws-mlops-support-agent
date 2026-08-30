"""Tests for aws_mlops_support_agent.settings.

The important one is the DRY_RUN parse: it is the only thing standing between
a config typo and a real Jira ticket, so it must fail SAFE.
"""

import pytest
from aws_mlops_support_agent import settings
from rag_core import config as rag_config

ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_CHAT_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "PINECONE_INDEX_METRIC",
    "AWS_REGION",
    "RAG_TOP_K",
    "RAG_MIN_TOP_SCORE",
    "JIRA_BASE_URL",
    "JIRA_EMAIL",
    "JIRA_API_TOKEN",
    "JIRA_PROJECT_KEY",
    "DRY_RUN",
)


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    """Keep tests independent of the developer's real .env / environment."""
    monkeypatch.setattr(rag_config, "load_dotenv", lambda: None)
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")


# --- the DRY_RUN safety gate ---


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("false", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        (" false ", False),  # surrounding whitespace still disables
        ("true", True),
        ("flase", True),  # typo must fail safe (stay in dry-run)
        ("", True),
    ],
)
def test_dry_run_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("DRY_RUN", raw)
    assert settings.load_settings().dry_run is expected


def test_dry_run_defaults_to_true_when_unset():
    """The safety default: no DRY_RUN in the environment means dry-run ON."""
    assert settings.load_settings().dry_run is True


def test_force_dry_run_overrides_a_live_config(monkeypatch):
    monkeypatch.setenv("DRY_RUN", "false")
    cfg = settings.load_settings()
    assert cfg.dry_run is False
    assert settings.force_dry_run(cfg).dry_run is True


# --- Jira fields ---


def test_jira_vars_optional_and_default_to_none():
    cfg = settings.load_settings()
    assert cfg.jira_base_url is None
    assert cfg.jira_email is None
    assert cfg.jira_api_token is None
    assert cfg.jira_project_key is None


def test_jira_vars_read_from_environment(monkeypatch):
    monkeypatch.setenv("JIRA_BASE_URL", "https://example.atlassian.net")
    monkeypatch.setenv("JIRA_EMAIL", "me@example.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "secret-token")
    monkeypatch.setenv("JIRA_PROJECT_KEY", "AS")
    cfg = settings.load_settings()
    assert cfg.jira_base_url == "https://example.atlassian.net"
    assert cfg.jira_email == "me@example.com"
    assert cfg.jira_api_token == "secret-token"
    assert cfg.jira_project_key == "AS"


def test_empty_jira_var_becomes_none(monkeypatch):
    """A blank value in .env means 'not configured', not an empty string."""
    monkeypatch.setenv("JIRA_BASE_URL", "")
    assert settings.load_settings().jira_base_url is None


# --- the wrapped RagConfig comes from this project's committed config.yml ---


def test_rag_config_loaded_from_project_config_yml():
    rag = settings.load_settings().rag
    assert rag.project == "aws-mlops-support-agent"
    assert rag.pinecone_index_name == "aws-mlops-docs"
    assert rag.retrieval.min_top_score == 0.35
    assert {s.id for s in rag.sources} == {"codebuild", "codepipeline"}


def test_env_still_overrides_the_project_config_yml(monkeypatch):
    monkeypatch.setenv("PINECONE_INDEX_NAME", "override-index")
    assert settings.load_settings().rag.pinecone_index_name == "override-index"

"""This project's settings: the generic RagConfig plus what only we need.

`rag_core.config.RagConfig` covers everything the RAG engine cares about
(embeddings, llm, vectorstore, splitter, retriever, generation, sources).
The two things it must NOT know about are the Jira credentials and the
`DRY_RUN` safety gate — those are specific to this project's escalation
workflow, so they live here.

`AgentConfig` wraps rather than subclasses RagConfig: the engine keeps its own
frozen type, and `cfg.rag` is what gets handed to every rag_core function.

Verify manually with:  uv run python -m aws_mlops_support_agent.settings
"""

import os
from dataclasses import dataclass, replace
from pathlib import Path

from rag_core.config import RagConfig, describe, load_config

# The committed, non-secret corpus config that ships inside this package.
CONFIG_PATH = Path(__file__).parent / "config.yml"


def _parse_dry_run(value: str | None) -> bool:
    """Fail-safe: only an explicit false/0/no turns dry-run OFF.

    Unset, empty, or a typo like "flase" all stay True, so a config mistake
    can never cause real Jira tickets to be created.
    """
    if value is None:
        return True
    return value.strip().lower() not in ("false", "0", "no")


@dataclass(frozen=True)
class AgentConfig:
    # Everything the RAG engine needs.
    rag: RagConfig
    # Jira (optional — the agent shows a draft instead of crashing when unset;
    # jira_tool validates these itself when it is actually used).
    jira_base_url: str | None
    jira_email: str | None
    jira_api_token: str | None
    jira_project_key: str | None
    # Safety gate on real Jira ticket creation. Defaults to True.
    dry_run: bool


def load_settings(config_path: Path | str | None = None) -> AgentConfig:
    """Load this project's config.yml + the environment into an AgentConfig."""
    # load_config also calls load_dotenv(), so the Jira vars read below are
    # already in os.environ by the time we get here.
    rag = load_config(config_path or CONFIG_PATH)
    return AgentConfig(
        rag=rag,
        jira_base_url=os.environ.get("JIRA_BASE_URL") or None,
        jira_email=os.environ.get("JIRA_EMAIL") or None,
        jira_api_token=os.environ.get("JIRA_API_TOKEN") or None,
        jira_project_key=os.environ.get("JIRA_PROJECT_KEY") or None,
        dry_run=_parse_dry_run(os.environ.get("DRY_RUN")),
    )


def force_dry_run(cfg: AgentConfig) -> AgentConfig:
    """Return a copy that can never create a real Jira ticket.

    Used by the public demo, where DRY_RUN must not be defeatable by whatever
    happens to be set in the environment.
    """
    return replace(cfg, dry_run=True)


if __name__ == "__main__":
    # Manual sanity check. Secrets are never printed — only whether they're set.
    cfg = load_settings()
    print(describe(cfg.rag))
    print(f"jira_base_url = {cfg.jira_base_url}")
    print(f"jira_email = {cfg.jira_email}")
    print(f"jira_api_token = {'(set)' if cfg.jira_api_token else '(not set)'}")
    print(f"jira_project_key = {cfg.jira_project_key}")
    print(f"dry_run = {cfg.dry_run}")

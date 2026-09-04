"""Shared test helpers for the agent package.

`make_config` builds an AgentConfig without touching the environment or
config.yml, so graph/Jira tests stay fully offline. The nested RagConfig is
filled with placeholders: tests inject fake retrievers and answerers, so no
real model or index name is ever read.
"""

import sys
from pathlib import Path

# Under pytest's importlib import mode (needed at the repo root: this dir has
# no __init__.py, and rag_bench_eval/tests also has a test_settings.py, which
# the default "prepend" mode can't disambiguate by basename alone), sibling
# test modules can no longer do a bare `from conftest import make_config` —
# importlib mode doesn't add a test file's own directory to sys.path. Adding
# it here, in the one file pytest always imports regardless of mode, keeps
# that import working without hacking every test file that uses it.
sys.path.insert(0, str(Path(__file__).parent))

from aws_mlops_support_agent.settings import AgentConfig
from rag_core.config import RagConfig
from rag_core.config.pipeline_parts import GenerationConfig, RetrieverConfig, SplitterConfig
from rag_core.config.providers import EmbeddingConfig, LLMConfig, VectorStoreConfig


def make_rag_config(**overrides) -> RagConfig:
    fields = {
        "embeddings": EmbeddingConfig(),
        "llm": LLMConfig(),
        "vectorstore": VectorStoreConfig(),
        "splitter": SplitterConfig(),
        # This project's own config.yml sets min_top_score: 0.35 (not
        # RetrieverConfig's generic 0.675 default) — match it here so graph/
        # ticket tests' score fixtures test against the real threshold.
        "retriever": RetrieverConfig(min_top_score=0.35),
        "generation": GenerationConfig(),
        "sources": (),
        "loader_extensions": (),
    }
    fields.update(overrides)
    return RagConfig(**fields)


def make_config(**overrides) -> AgentConfig:
    """Build an AgentConfig.

    Jira/dry_run fields are set directly; anything else is treated as a
    RagConfig field, so `make_config(retrieval=RetrievalConfig(top_k=2))`
    works without spelling out the whole nested object.
    """
    agent_fields = ("jira_base_url", "jira_email", "jira_api_token", "jira_project_key", "dry_run")
    rag_overrides = {k: overrides.pop(k) for k in list(overrides) if k not in agent_fields}
    fields = {
        "rag": make_rag_config(**rag_overrides),
        "jira_base_url": None,
        "jira_email": None,
        "jira_api_token": None,
        "jira_project_key": None,
        "dry_run": True,
    }
    fields.update(overrides)
    return AgentConfig(**fields)

"""Shared test helpers for the agent package.

`make_config` builds an AgentConfig without touching the environment or
config.yml, so graph/Jira tests stay fully offline. The nested RagConfig is
filled with placeholders: tests inject fake retrievers and answerers, so no
real model or index name is ever read.
"""

from aws_mlops_support_agent.settings import AgentConfig
from rag_core.config import ChunkingConfig, RagConfig, RetrievalConfig


def make_rag_config(**overrides) -> RagConfig:
    fields = {
        "openai_api_key": "fake",
        "openai_chat_model": "fake-model",
        "openai_embedding_model": "fake-embed",
        "pinecone_api_key": "fake",
        "pinecone_index_name": "fake-index",
        "pinecone_index_metric": "cosine",
        "aws_region": "us-east-1",
        "project": "test-project",
        "chunking": ChunkingConfig(),
        "retrieval": RetrievalConfig(),
        "sources": (),
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

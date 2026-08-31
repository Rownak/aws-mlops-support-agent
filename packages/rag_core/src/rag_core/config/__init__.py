"""
Configuration for the rag_core package.

Two layers:

- ``Config`` (``config.loader``) loads raw YAML and resolves ``${VAR}``
  placeholders against the environment. This is ragwire's original loader,
  kept for callers that want free-form dot-notation access to settings the
  typed tree doesn't know about (e.g. a source's ``aws_access_key_id``).
- ``RagConfig`` (``config.root``) sits on top of it and is what ``RagCore``
  actually uses: a frozen dataclass tree, one field per setting that has a
  well-known name and a sensible default, resolved with **env var >
  config.yaml > built-in default** precedence. Secrets (API keys) are read
  ONLY from the environment, never from config.yaml, and missing ones
  required by the configured provider are collected and reported together
  rather than failing one at a time.

``RagConfig``'s provider blocks expose ``.as_dict()`` so the factories
(``embeddings.factory.get_embedding``, ``llm.factory.get_llm``) and
``vectorstores.pinecone_store.PineconeStore`` keep taking a plain dict —
nothing downstream of config parsing is coupled to these types.

Module layout, lowest layer first:

    env.py             provider -> API-key env maps, the precedence helper
    base.py            DictLike mixin (dict rendering + secret masking)
    loader.py          Config: raw YAML + ${VAR} substitution
    providers.py       EmbeddingConfig, LLMConfig, VectorStoreConfig
    pipeline_parts.py  SplitterConfig, RetrieverConfig, GenerationConfig, SourceSpec
    root.py            RagConfig, load_config, describe

Everything public is re-exported here, so ``from rag_core.config import X``
keeps working regardless of which module X lives in.
"""

from .loader import Config
from .pipeline_parts import (
    GenerationConfig,
    RetrieverConfig,
    SourceSpec,
    SplitterConfig,
)
from .providers import EmbeddingConfig, LLMConfig, VectorStoreConfig
from .root import RagConfig, describe, load_config

__all__ = [
    "Config",
    "RagConfig",
    "load_config",
    "describe",
    "EmbeddingConfig",
    "LLMConfig",
    "VectorStoreConfig",
    "SplitterConfig",
    "RetrieverConfig",
    "GenerationConfig",
    "SourceSpec",
]

"""Config blocks for the fixed pipeline stages: splitting, retrieval, generation, sources.

Unlike `providers.py`, none of these select a backend or carry a secret —
they are plain tuning values, so they need no provider map and no
`missing_secrets()`.
"""

import os
from dataclasses import dataclass, field

from .env import _env_or

#: Kept here rather than imported from `retriever.confidence` (which defines
#: the same default) because `retriever` imports this module — importing back
#: would be a cycle. The two must stay in step; see that module for the scale.
DEFAULT_MIN_TOP_SCORE = 0.675


@dataclass(frozen=True)
class SplitterConfig:
    chunk_size: int = 800
    chunk_overlap: int = 100
    strategy: str = "markdown"

    @classmethod
    def from_raw(cls, raw: dict) -> "SplitterConfig":
        """Parse the ``splitter`` block, env > yaml > default per field."""
        return cls(
            chunk_size=int(_env_or("RAG_CHUNK_SIZE", raw.get("chunk_size"), cls.chunk_size)),
            chunk_overlap=int(
                _env_or("RAG_CHUNK_OVERLAP", raw.get("chunk_overlap"), cls.chunk_overlap)
            ),
            strategy=raw.get("strategy") or cls.strategy,
        )


@dataclass(frozen=True)
class RetrieverConfig:
    search_type: str = "similarity"
    top_k: int = 5
    #: On the normalized [0, 1] relevance scale (see retriever.confidence).
    #: None disables the confidence check.
    min_top_score: float | None = DEFAULT_MIN_TOP_SCORE
    # The retriever.rerank sub-block, kept as a plain dict — its shape is
    # provider-defined (cross_encoder vs cohere take different keys) and
    # get_reranker()/resolve_fetch_k() already accept a dict or None.
    rerank: dict | None = None

    @classmethod
    def from_raw(cls, raw: dict) -> "RetrieverConfig":
        """Parse the ``retriever`` block, env > yaml > default per field."""
        return cls(
            search_type=raw.get("search_type") or cls.search_type,
            top_k=int(_env_or("RAG_TOP_K", raw.get("top_k"), cls.top_k)),
            min_top_score=cls._parse_min_top_score(raw),
            rerank=raw.get("rerank"),
        )

    @staticmethod
    def _parse_min_top_score(raw: dict) -> float | None:
        """Resolve min_top_score, keeping an explicit YAML `null` as None.

        Not via `_env_or`: it coalesces falsy values, so an explicit `null`
        (meaning "disable the check") would silently fall back to the default
        instead. `key in raw` is what distinguishes "set to null" from "absent".
        """
        env = os.environ.get("RAG_MIN_TOP_SCORE")
        if env:
            return None if env.lower() in ("none", "null") else float(env)
        if "min_top_score" in raw:
            value = raw["min_top_score"]
            return None if value is None else float(value)
        return DEFAULT_MIN_TOP_SCORE


@dataclass(frozen=True)
class GenerationConfig:
    max_context_chars: int = 12000
    system_prompt: str | None = None

    @classmethod
    def from_raw(cls, raw: dict) -> "GenerationConfig":
        """Parse the ``generation`` block. No env overrides for these."""
        return cls(
            max_context_chars=int(raw.get("max_context_chars") or cls.max_context_chars),
            system_prompt=raw.get("system_prompt"),
        )


@dataclass(frozen=True)
class SourceSpec:
    """One entry of the ``sources`` config block, kept open-ended.

    Every source type (local, s3, a project's own registered type) has its
    own keys, so this is intentionally a thin pass-through rather than a
    per-type dataclass — `build_source()` already validates `type` and
    dispatches on it.
    """

    options: dict = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict) -> "SourceSpec":
        """Wrap one ``sources`` entry verbatim, `type` key included."""
        return cls(options=dict(raw))

    def as_dict(self) -> dict:
        return dict(self.options)

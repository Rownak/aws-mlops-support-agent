"""Config blocks for the fixed pipeline stages: splitting, retrieval, generation, sources.

Unlike `providers.py`, none of these select a backend or carry a secret —
they are plain tuning values, so they need no provider map and no
`missing_secrets()`.
"""

from dataclasses import dataclass, field

from .env import _env_or


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
    min_top_score: float = 0.35
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
            min_top_score=float(
                _env_or("RAG_MIN_TOP_SCORE", raw.get("min_top_score"), cls.min_top_score)
            ),
            rerank=raw.get("rerank"),
        )


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

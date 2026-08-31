"""The typed config tree: `RagConfig`, its loader, and its human-readable renderer.

`RagConfig` composes the leaf blocks from `providers` and `pipeline_parts`,
applying **env var > config.yaml > built-in default** precedence to each
field and failing fast — with every missing secret reported at once — rather
than one error per run.
"""

from dataclasses import dataclass, fields

from .loader import Config
from .pipeline_parts import (
    GenerationConfig,
    RetrieverConfig,
    SourceSpec,
    SplitterConfig,
)
from .providers import EmbeddingConfig, LLMConfig, VectorStoreConfig

#: Fallback when config.yaml has no ``loader.extensions``.
DEFAULT_LOADER_EXTENSIONS = (".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md")


@dataclass(frozen=True)
class RagConfig:
    embeddings: EmbeddingConfig
    llm: LLMConfig
    vectorstore: VectorStoreConfig
    splitter: SplitterConfig
    retriever: RetrieverConfig
    generation: GenerationConfig
    sources: tuple[SourceSpec, ...]
    loader_extensions: tuple[str, ...]

    @staticmethod
    def from_config(config: "Config") -> "RagConfig":
        """Build a frozen RagConfig from a loaded Config (config.yaml + env).

        Each block parses itself: adding a provider means editing that block's
        class in `providers.py`, not this method.
        """
        cfg = RagConfig(
            embeddings=EmbeddingConfig.from_raw(config.get("embeddings", {}) or {}),
            llm=LLMConfig.from_raw(config.get("llm", {}) or {}),
            vectorstore=VectorStoreConfig.from_raw(config.get("vectorstore", {}) or {}),
            splitter=SplitterConfig.from_raw(config.get("splitter", {}) or {}),
            retriever=RetrieverConfig.from_raw(config.get("retriever", {}) or {}),
            generation=GenerationConfig.from_raw(config.get("generation", {}) or {}),
            sources=tuple(
                SourceSpec.from_raw(entry) for entry in (config.get("sources", []) or [])
            ),
            loader_extensions=tuple(
                (config.get("loader", {}) or {}).get("extensions", DEFAULT_LOADER_EXTENSIONS)
            ),
        )
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        """Fail fast, reporting every missing required secret in one error.

        Each block decides what it requires; blocks that carry no secret
        (splitter, retriever, generation, sources) simply don't define
        `missing_secrets`. A provider that needs no key — ollama, or a local
        Pinecone reached via `vectorstore.host` — reports nothing missing,
        because demanding a key nothing reads would make local dev impossible.
        """
        missing: list[str] = []
        for f in fields(self):
            block = getattr(self, f.name)
            reporter = getattr(block, "missing_secrets", None)
            if reporter is not None:
                missing.extend(reporter())

        if missing:
            raise RuntimeError(
                "Missing required environment variables: " + "; ".join(missing) + ". "
                "Copy .env.example to .env and fill them in."
            )


def load_config(path: str) -> RagConfig:
    """Load config.yaml and the environment into a validated RagConfig.

    Example:
        >>> cfg = load_config("config.yaml")  # doctest: +SKIP
        >>> cfg.retriever.top_k  # doctest: +SKIP
        5
    """
    return RagConfig.from_config(Config(path))


def describe(cfg: RagConfig) -> str:
    """Render a config for manual inspection, with secrets masked.

    Which fields count as secret is read off each block's own
    ``_secret_fields`` (see `providers.py`), so a new secret-bearing block
    masks correctly without editing anything here.
    """
    lines = []
    for f in fields(cfg):
        value = getattr(cfg, f.name)
        # Secrets are never printed — only whether they are set.
        secret_fields = getattr(type(value), "_secret_fields", None)
        if secret_fields and hasattr(value, "__dataclass_fields__"):
            parts = []
            for nested in fields(value):
                nested_value = getattr(value, nested.name)
                if nested.name in secret_fields:
                    nested_value = "(set)" if nested_value else "(not set)"
                parts.append(f"{nested.name}={nested_value}")
            lines.append(f"{f.name} = {{{', '.join(parts)}}}")
        else:
            lines.append(f"{f.name} = {value}")
    return "\n".join(lines)

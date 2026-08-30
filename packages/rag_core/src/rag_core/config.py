"""Load and validate the generic RAG engine's configuration.

Two layers, deliberately different in kind (design.md §5):

- **Secrets** come only from the environment — locally via a gitignored
  `.env`, in prod from AWS Secrets Manager. They are never read from a file.
- **Corpus-shaped settings** (index name, models, chunking, retrieval) come
  from the project's committed `config.yml`, with environment variables able
  to override any of them.

Precedence: **env var > config.yml > built-in default.**

Nothing here knows about AWS docs, Jira, or any particular corpus — that is
the project package's job. `load_config()` fails fast, reporting ALL missing
required variables in one pass so they can be fixed in one go.
"""

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml
from dotenv import load_dotenv

# The engine can't do anything useful without these two. Everything else has
# a default, comes from config.yml, or belongs to a project package.
REQUIRED_VARS = ["OPENAI_API_KEY", "PINECONE_API_KEY"]

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_INDEX_NAME = "rag-docs"
DEFAULT_INDEX_METRIC = "cosine"  # standard choice for OpenAI embeddings
DEFAULT_AWS_REGION = "us-east-1"
DEFAULT_CHUNK_SIZE_TOKENS = 800
DEFAULT_CHUNK_OVERLAP_TOKENS = 100
DEFAULT_TOP_K = 4
DEFAULT_MIN_TOP_SCORE = 0.35


@dataclass(frozen=True)
class ChunkingConfig:
    # ~800 tokens sits mid-range of the 500-1000 target; 100 tokens ~ 12%
    # overlap so a sentence cut at a boundary still appears whole somewhere.
    size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS
    # Regexes stripped from raw markdown before splitting. Corpus-specific
    # cruft (e.g. awsdocs' `<a name="...">` heading anchors) is configured
    # here rather than hardcoded in the engine.
    strip_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = DEFAULT_TOP_K
    # Cosine similarity is not a probability: a usable threshold depends on
    # BOTH the embedding model and the corpus, so it must be per-project.
    min_top_score: float = DEFAULT_MIN_TOP_SCORE


@dataclass(frozen=True)
class SourceSpec:
    """One document source, as declared in config.yml (see sources.py)."""

    # Short slug used in chunk metadata, chunk IDs, and local paths.
    id: str
    # Which project-side adapter fetches this source.
    loader: str
    # Everything else the adapter needs (git_url, docs_base_url, ...). Kept
    # open-ended so rag_core never has to learn a new corpus's vocabulary.
    options: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RagConfig:
    # OpenAI (LLM + embeddings). The embedding model must be identical at
    # ingestion time and query time, or retrieval silently degrades — that's
    # why it lives in config and is never hardcoded in the ingest script.
    openai_api_key: str
    openai_chat_model: str
    openai_embedding_model: str
    # Pinecone (vector DB)
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_index_metric: str
    # Cloud region the serverless index lives in.
    aws_region: str
    # Name of the project this config belongs to; used in log lines only.
    project: str
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    sources: tuple[SourceSpec, ...]


def _load_yaml(path: Path | str | None) -> dict:
    """Read config.yml, or return {} when no path was given.

    A path that WAS given must exist and must parse to a mapping — a typo in
    the path would otherwise silently fall back to defaults and quietly build
    the wrong index.
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Config file {path} is not valid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Config file {path} must contain a mapping at the top level.")
    return data


def _env_or(name: str, yaml_value, default):
    """Apply the precedence rule: env var > config.yml > built-in default.

    `os.environ.get(name) or ...` (not `.get(name, ...)`) so that an empty
    string in .env still falls through to the next layer.
    """
    return os.environ.get(name) or yaml_value or default


def _parse_sources(raw) -> tuple[SourceSpec, ...]:
    """Turn the config.yml `sources:` list into SourceSpecs.

    Every key other than id/loader is kept verbatim in `options` and handed
    to the project's adapter untouched.
    """
    if not raw:
        return ()
    if not isinstance(raw, list):
        raise RuntimeError("config.yml: 'sources' must be a list.")
    specs = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise RuntimeError("config.yml: every entry under 'sources' must be a mapping.")
        missing = [key for key in ("id", "loader") if not entry.get(key)]
        if missing:
            raise RuntimeError(
                f"config.yml: source entry {entry!r} is missing: {', '.join(missing)}."
            )
        options = {k: v for k, v in entry.items() if k not in ("id", "loader")}
        specs.append(SourceSpec(id=entry["id"], loader=entry["loader"], options=options))
    return tuple(specs)


def load_config(path: Path | str | None = None) -> RagConfig:
    """Build a frozen RagConfig from config.yml (optional) + the environment."""
    # Reads .env into os.environ; a no-op if the file is absent (e.g. in CI,
    # where vars come from the environment directly). Does not override
    # variables that are already set.
    load_dotenv()

    data = _load_yaml(path)
    index = data.get("index") or {}
    models = data.get("models") or {}
    chunking = data.get("chunking") or {}
    retrieval = data.get("retrieval") or {}

    # Collect ALL missing vars before failing, so the user fixes them in one
    # pass instead of replaying error-by-error.
    # (Empty string counts as missing — .env.example ships blank values.)
    missing = [name for name in REQUIRED_VARS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )

    return RagConfig(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_chat_model=_env_or("OPENAI_CHAT_MODEL", models.get("chat"), DEFAULT_CHAT_MODEL),
        openai_embedding_model=_env_or(
            "OPENAI_EMBEDDING_MODEL", models.get("embedding"), DEFAULT_EMBEDDING_MODEL
        ),
        pinecone_api_key=os.environ["PINECONE_API_KEY"],
        pinecone_index_name=_env_or("PINECONE_INDEX_NAME", index.get("name"), DEFAULT_INDEX_NAME),
        pinecone_index_metric=_env_or(
            "PINECONE_INDEX_METRIC", index.get("metric"), DEFAULT_INDEX_METRIC
        ),
        aws_region=_env_or("AWS_REGION", index.get("region"), DEFAULT_AWS_REGION),
        project=data.get("project") or "rag",
        chunking=ChunkingConfig(
            size_tokens=int(
                _env_or(
                    "RAG_CHUNK_SIZE_TOKENS", chunking.get("size_tokens"), DEFAULT_CHUNK_SIZE_TOKENS
                )
            ),
            overlap_tokens=int(
                _env_or(
                    "RAG_CHUNK_OVERLAP_TOKENS",
                    chunking.get("overlap_tokens"),
                    DEFAULT_CHUNK_OVERLAP_TOKENS,
                )
            ),
            # Patterns are a list, so they come from YAML only — there is no
            # sensible way to express a regex list in a single env var.
            strip_patterns=tuple(chunking.get("strip_patterns") or ()),
        ),
        retrieval=RetrievalConfig(
            top_k=int(_env_or("RAG_TOP_K", retrieval.get("top_k"), DEFAULT_TOP_K)),
            min_top_score=float(
                _env_or("RAG_MIN_TOP_SCORE", retrieval.get("min_top_score"), DEFAULT_MIN_TOP_SCORE)
            ),
        ),
        sources=_parse_sources(data.get("sources")),
    )


# Secrets are never printed — only whether they are set.
SECRET_FIELDS = {"openai_api_key", "pinecone_api_key"}


def describe(cfg: RagConfig) -> str:
    """Render a config for manual inspection, with secrets masked."""
    lines = []
    for f in fields(cfg):
        value = getattr(cfg, f.name)
        if f.name in SECRET_FIELDS:
            value = "(set)" if value else "(not set)"
        lines.append(f"{f.name} = {value}")
    return "\n".join(lines)

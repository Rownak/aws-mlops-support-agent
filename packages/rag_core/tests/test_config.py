"""Tests for rag_core.config — run with: uv run pytest

Covers the two behaviors the engine's config must never lose:
  1. fail-fast: every missing required var reported in ONE error;
  2. precedence: env var > config.yml > built-in default.
"""

import pytest
from rag_core import config

# Every variable load_config reads. Cleared before each test so a developer's
# real environment can never change a result.
ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_CHAT_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "PINECONE_INDEX_METRIC",
    "AWS_REGION",
    "RAG_CHUNK_SIZE_TOKENS",
    "RAG_CHUNK_OVERLAP_TOKENS",
    "RAG_TOP_K",
    "RAG_MIN_TOP_SCORE",
)

# A config.yml exercising every section, used by the precedence tests.
SAMPLE_YAML = """
project: sample-project
index:
  name: yaml-index
  metric: dotproduct
  region: eu-west-1
models:
  embedding: text-embedding-3-large
  chat: gpt-4o
chunking:
  size_tokens: 500
  overlap_tokens: 50
  strip_patterns: ['<a name="[^"]*"></a>']
retrieval:
  top_k: 7
  min_top_score: 0.5
sources:
  - id: codebuild
    loader: awsdocs_git
    git_url: https://example.invalid/codebuild.git
    docs_base_url: https://docs.example.invalid/codebuild/
"""


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    """Disable .env loading, clear every read var, set the two required ones."""
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")


@pytest.fixture
def config_file(tmp_path):
    """Write SAMPLE_YAML to a temp file and return its path."""
    path = tmp_path / "config.yml"
    path.write_text(SAMPLE_YAML, encoding="utf-8")
    return path


# --- fail-fast on missing secrets (ported verbatim from src/config.py) ---


def test_missing_required_var_fails_with_its_name(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        config.load_config()


def test_all_missing_vars_reported_at_once(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("PINECONE_API_KEY", "")  # empty counts as missing
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY, PINECONE_API_KEY"):
        config.load_config()


# --- precedence level 3: built-in defaults ---


def test_defaults_applied_when_no_yaml_and_no_env():
    cfg = config.load_config()
    assert cfg.openai_chat_model == config.DEFAULT_CHAT_MODEL
    assert cfg.openai_embedding_model == config.DEFAULT_EMBEDDING_MODEL
    assert cfg.pinecone_index_name == config.DEFAULT_INDEX_NAME
    assert cfg.pinecone_index_metric == config.DEFAULT_INDEX_METRIC
    assert cfg.aws_region == config.DEFAULT_AWS_REGION
    assert cfg.chunking.size_tokens == config.DEFAULT_CHUNK_SIZE_TOKENS
    assert cfg.chunking.overlap_tokens == config.DEFAULT_CHUNK_OVERLAP_TOKENS
    assert cfg.chunking.strip_patterns == ()
    assert cfg.retrieval.top_k == config.DEFAULT_TOP_K
    assert cfg.retrieval.min_top_score == config.DEFAULT_MIN_TOP_SCORE
    assert cfg.sources == ()


def test_empty_env_value_falls_back_to_default(monkeypatch):
    """An empty string in .env must not win over the default."""
    monkeypatch.setenv("PINECONE_INDEX_NAME", "")
    assert config.load_config().pinecone_index_name == config.DEFAULT_INDEX_NAME


# --- precedence level 2: config.yml beats the default ---


def test_yaml_overrides_defaults(config_file):
    cfg = config.load_config(config_file)
    assert cfg.project == "sample-project"
    assert cfg.pinecone_index_name == "yaml-index"
    assert cfg.pinecone_index_metric == "dotproduct"
    assert cfg.aws_region == "eu-west-1"
    assert cfg.openai_chat_model == "gpt-4o"
    assert cfg.openai_embedding_model == "text-embedding-3-large"
    assert cfg.chunking.size_tokens == 500
    assert cfg.chunking.overlap_tokens == 50
    assert cfg.chunking.strip_patterns == ('<a name="[^"]*"></a>',)
    assert cfg.retrieval.top_k == 7
    assert cfg.retrieval.min_top_score == 0.5


def test_yaml_sources_parsed_into_specs(config_file):
    (source,) = config.load_config(config_file).sources
    assert source.id == "codebuild"
    assert source.loader == "awsdocs_git"
    # Adapter-specific keys stay in options, untouched by rag_core.
    assert source.options == {
        "git_url": "https://example.invalid/codebuild.git",
        "docs_base_url": "https://docs.example.invalid/codebuild/",
    }


# --- precedence level 1: env var beats config.yml ---


def test_env_overrides_yaml(monkeypatch, config_file):
    monkeypatch.setenv("PINECONE_INDEX_NAME", "env-index")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("RAG_TOP_K", "2")
    monkeypatch.setenv("RAG_MIN_TOP_SCORE", "0.9")
    monkeypatch.setenv("RAG_CHUNK_SIZE_TOKENS", "123")
    cfg = config.load_config(config_file)
    assert cfg.pinecone_index_name == "env-index"
    assert cfg.openai_chat_model == "gpt-4o-mini"
    assert cfg.retrieval.top_k == 2
    assert cfg.retrieval.min_top_score == 0.9
    assert cfg.chunking.size_tokens == 123
    # Untouched YAML values still come through.
    assert cfg.chunking.overlap_tokens == 50


def test_numeric_env_overrides_are_typed(monkeypatch, config_file):
    """Env vars are strings; the config must coerce them, not leak str."""
    monkeypatch.setenv("RAG_TOP_K", "9")
    monkeypatch.setenv("RAG_MIN_TOP_SCORE", "0.25")
    cfg = config.load_config(config_file)
    assert cfg.retrieval.top_k == 9
    assert isinstance(cfg.retrieval.min_top_score, float)


# --- a bad config file must fail clearly, not silently use defaults ---


def test_missing_config_file_fails(tmp_path):
    with pytest.raises(RuntimeError, match="Config file not found"):
        config.load_config(tmp_path / "nope.yml")


def test_malformed_yaml_fails(tmp_path):
    path = tmp_path / "bad.yml"
    path.write_text("index: [unclosed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not valid YAML"):
        config.load_config(path)


def test_non_mapping_yaml_fails(tmp_path):
    path = tmp_path / "list.yml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="mapping at the top level"):
        config.load_config(path)


def test_empty_yaml_file_is_allowed(tmp_path):
    """An empty config.yml is valid — everything falls back to defaults."""
    path = tmp_path / "empty.yml"
    path.write_text("", encoding="utf-8")
    assert config.load_config(path).pinecone_index_name == config.DEFAULT_INDEX_NAME


def test_source_missing_loader_fails(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text("sources:\n  - id: codebuild\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="loader"):
        config.load_config(path)


def test_describe_masks_secrets():
    text = config.describe(config.load_config())
    assert "test-openai-key" not in text
    assert "test-pinecone-key" not in text
    assert "openai_api_key = (set)" in text

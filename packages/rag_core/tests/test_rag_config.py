"""Tests for RagConfig: typed, validated, precedence-resolved configuration."""

import pytest

from rag_core.config import load_config

# Config() calls load_dotenv() on every load, which repopulates os.environ
# from the workspace .env even after monkeypatch.delenv() clears it. Tests
# that need a key genuinely absent must also point HOME/cwd's .env away, so
# instead we neutralize dotenv for these tests specifically.
#
# The target is config.loader (where Config lives and resolves the name), not
# the config package — patching the package attribute would silently no-op.


@pytest.fixture(autouse=True)
def _no_dotenv_autoload(monkeypatch):
    monkeypatch.setattr("rag_core.config.loader.load_dotenv", lambda *a, **k: None)


def _write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


# --- defaults, when config.yaml omits a block entirely ---


def test_defaults_apply_when_config_is_minimal(tmp_path, monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    path = _write(tmp_path, "vectorstore:\n  host: \"http://localhost:5080\"\n")

    cfg = load_config(path)

    assert cfg.embeddings.provider == "ollama"
    assert cfg.embeddings.model == "nomic-embed-text"
    assert cfg.llm.provider == "ollama"
    assert cfg.splitter.chunk_size == 800
    assert cfg.splitter.strategy == "markdown"
    assert cfg.retriever.top_k == 5
    assert cfg.retriever.min_top_score == 0.35
    assert cfg.generation.max_context_chars == 12000
    assert cfg.sources == ()


# --- precedence: env var > yaml > default ---


def test_yaml_value_overrides_default(tmp_path, monkeypatch):
    monkeypatch.delenv("RAG_TOP_K", raising=False)
    path = _write(tmp_path, "vectorstore:\n  host: \"http://localhost:5080\"\nretriever:\n  top_k: 8\n")

    cfg = load_config(path)
    assert cfg.retriever.top_k == 8


def test_env_var_overrides_yaml_value(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "12")
    path = _write(tmp_path, "vectorstore:\n  host: \"http://localhost:5080\"\nretriever:\n  top_k: 8\n")

    cfg = load_config(path)
    assert cfg.retriever.top_k == 12


def test_env_var_applies_with_no_yaml_value_present(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_MIN_TOP_SCORE", "0.5")
    path = _write(tmp_path, "vectorstore:\n  host: \"http://localhost:5080\"\n")

    cfg = load_config(path)
    assert cfg.retriever.min_top_score == 0.5


# --- secrets: env-only, never read from yaml as a value the user set there ---


def test_pinecone_api_key_is_read_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "secret-value")
    path = _write(tmp_path, "vectorstore:\n  collection_name: idx\n")

    cfg = load_config(path)
    assert cfg.vectorstore.api_key == "secret-value"


def test_openai_key_is_read_from_env_for_openai_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "pc-key")
    monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
    path = _write(tmp_path, "embeddings:\n  provider: openai\nllm:\n  provider: openai\n")

    cfg = load_config(path)
    assert cfg.embeddings.api_key == "oa-key"
    assert cfg.llm.api_key == "oa-key"


# --- validation: missing required secrets are collected, not raised one at a time ---


def test_missing_pinecone_key_raises_when_using_managed_service(tmp_path, monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    path = _write(tmp_path, "vectorstore:\n  collection_name: idx\n")  # no host -> managed service

    with pytest.raises(RuntimeError, match="PINECONE_API_KEY"):
        load_config(path)


def test_missing_pinecone_key_is_fine_with_local_host(tmp_path, monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    path = _write(tmp_path, "vectorstore:\n  host: \"http://localhost:5080\"\n")

    cfg = load_config(path)  # must not raise
    assert cfg.vectorstore.host == "http://localhost:5080"


def test_ollama_needs_no_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = _write(
        tmp_path,
        "vectorstore:\n  host: \"http://localhost:5080\"\n"
        "embeddings:\n  provider: ollama\nllm:\n  provider: ollama\n",
    )

    cfg = load_config(path)  # must not raise
    assert cfg.embeddings.api_key is None
    assert cfg.llm.api_key is None


def test_all_missing_secrets_are_reported_together(tmp_path, monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = _write(
        tmp_path,
        "embeddings:\n  provider: openai\nllm:\n  provider: openai\nvectorstore:\n  collection_name: idx\n",
    )

    with pytest.raises(RuntimeError) as exc_info:
        load_config(path)

    message = str(exc_info.value)
    assert "PINECONE_API_KEY" in message
    assert "OPENAI_API_KEY" in message


# --- sources round-trip through SourceSpec.as_dict() ---


def test_sources_round_trip_as_dict(tmp_path, monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    path = _write(
        tmp_path,
        "vectorstore:\n  host: \"http://localhost:5080\"\n"
        "sources:\n  - type: local\n    path: ./docs\n    recursive: true\n",
    )

    cfg = load_config(path)
    assert len(cfg.sources) == 1
    assert cfg.sources[0].as_dict() == {"type": "local", "path": "./docs", "recursive": True}


# --- as_dict() shapes feed straight into the existing factories ---


def test_embeddings_as_dict_has_provider_and_api_key_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    path = _write(tmp_path, "vectorstore:\n  host: \"http://localhost:5080\"\n")

    cfg = load_config(path)
    d = cfg.embeddings.as_dict()
    assert d["provider"] == "ollama"
    assert "api_key" in d


def test_vectorstore_as_dict_omits_host_when_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "k")
    path = _write(tmp_path, "vectorstore:\n  collection_name: idx\n")

    cfg = load_config(path)
    assert "host" not in cfg.vectorstore.as_dict()

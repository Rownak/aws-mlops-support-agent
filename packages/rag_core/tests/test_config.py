"""Tests for the dot-notation YAML + env config loader."""

import pytest

from rag_core.config import Config


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
vectorstore:
  collection_name: "my-index"
  use_sparse: false
retriever:
  top_k: 5
""",
        encoding="utf-8",
    )
    return str(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config(str(tmp_path / "nope.yaml"))


def test_dot_notation_get(config_path):
    cfg = Config(config_path)
    assert cfg.get("vectorstore.collection_name") == "my-index"
    assert cfg.get("retriever.top_k") == 5


def test_missing_key_returns_default(config_path):
    cfg = Config(config_path)
    assert cfg.get("nonexistent.key", "fallback") == "fallback"
    assert cfg.get("nonexistent.key") is None


def test_getitem_and_contains(config_path):
    cfg = Config(config_path)
    assert cfg["vectorstore.collection_name"] == "my-index"
    assert "vectorstore.collection_name" in cfg
    assert "nonexistent.key" not in cfg


def test_env_var_placeholder_is_resolved(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_SECRET_HOST", "http://example.invalid")
    path = tmp_path / "config.yaml"
    path.write_text('vectorstore:\n  host: "${MY_SECRET_HOST}"\n', encoding="utf-8")

    cfg = Config(str(path))
    assert cfg.get("vectorstore.host") == "http://example.invalid"


def test_unset_env_var_placeholder_is_left_untouched(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text('vectorstore:\n  host: "${NOT_SET_ANYWHERE}"\n', encoding="utf-8")

    cfg = Config(str(path))
    assert cfg.get("vectorstore.host") == "${NOT_SET_ANYWHERE}"

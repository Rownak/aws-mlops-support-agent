"""Tests for readiness preflight checks: live reachability, not config parsing."""

import httpx
import pytest

from rag_core.config.providers import EmbeddingConfig, LLMConfig, VectorStoreConfig
from rag_core.config.readiness import check_pinecone_local_ready, check_readiness


class _FakeResponse:
    def __init__(self, json_data=None, status_code=200):
        self._json = json_data or {}
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return self._json


def _tags_response(*models):
    return _FakeResponse({"models": [{"name": m} for m in models]})


# --- ollama ---


def test_ollama_ready_when_model_present(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _tags_response("qwen3:1.7b"))
    cfg = LLMConfig(provider="ollama", model="qwen3:1.7b")
    assert cfg.missing_readiness() == []


def test_ollama_reports_missing_model(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _tags_response("other-model"))
    cfg = LLMConfig(provider="ollama", model="qwen3:1.7b")
    problems = cfg.missing_readiness()
    assert len(problems) == 1
    assert "qwen3:1.7b" in problems[0]
    assert "ollama pull" in problems[0]


def test_ollama_reports_unreachable(monkeypatch):
    def _raise(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _raise)
    cfg = EmbeddingConfig(provider="ollama", base_url="http://localhost:11434")
    problems = cfg.missing_readiness()
    assert len(problems) == 1
    assert "http://localhost:11434" in problems[0]


# --- pinecone local ---


def test_pinecone_local_ready_when_reachable(monkeypatch):
    # A 404 (no root route) still proves something is listening.
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse(status_code=404))
    assert check_pinecone_local_ready("http://localhost:5080") == []


def test_pinecone_local_reports_unreachable(monkeypatch):
    def _raise(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _raise)
    problems = check_pinecone_local_ready("http://localhost:5080")
    assert len(problems) == 1
    assert "http://localhost:5080" in problems[0]
    assert "docker run" in problems[0]


# --- providers with no readiness concept skip the check entirely ---


def test_cloud_pinecone_skips_readiness_check(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "get", lambda *a, **k: calls.append(1))
    cfg = VectorStoreConfig(provider="pinecone", host=None, api_key="k")
    assert cfg.missing_readiness() == []
    assert calls == []


def test_openai_provider_skips_readiness_check(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "get", lambda *a, **k: calls.append(1))
    assert EmbeddingConfig(provider="openai", api_key="k").missing_readiness() == []
    assert LLMConfig(provider="openai", api_key="k").missing_readiness() == []
    assert calls == []


# --- aggregation ---


def test_check_readiness_aggregates_across_blocks(monkeypatch):
    def _raise(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _raise)
    monkeypatch.delenv("RAG_SKIP_READINESS_CHECK", raising=False)

    class _Config:
        embeddings = EmbeddingConfig(provider="ollama")
        llm = LLMConfig(provider="openai", api_key="k")  # no readiness concept
        vectorstore = VectorStoreConfig(provider="pinecone", host="http://localhost:5080")

    with pytest.raises(RuntimeError) as exc_info:
        check_readiness(_Config())

    message = str(exc_info.value)
    assert "Ollama" in message
    assert "Pinecone Local" in message


def test_skip_readiness_check_env_var(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "get", lambda *a, **k: calls.append(1))
    monkeypatch.setenv("RAG_SKIP_READINESS_CHECK", "1")

    class _Config:
        embeddings = EmbeddingConfig(provider="ollama")
        llm = LLMConfig(provider="ollama")
        vectorstore = VectorStoreConfig(provider="pinecone", host="http://localhost:5080")

    check_readiness(_Config())  # must not raise
    assert calls == []

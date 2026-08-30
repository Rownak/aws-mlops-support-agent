"""Tests for the Pinecone store's guards. No network — a fake client is injected.

The three failure modes worth protecting are all silent-corruption risks:
unknown model (guessing a dimension), dimension mismatch (unusable vectors),
and a missing index at query time (querying an empty index forever).
"""

import types

import pytest
from rag_core.config import ChunkingConfig, RagConfig, RetrievalConfig
from rag_core.vectorstore import pinecone_store


def make_cfg(embedding_model: str = "text-embedding-3-small") -> RagConfig:
    return RagConfig(
        openai_api_key="k",
        openai_chat_model="gpt-4o-mini",
        openai_embedding_model=embedding_model,
        pinecone_api_key="k",
        pinecone_index_name="test-index",
        pinecone_index_metric="cosine",
        aws_region="us-east-1",
        project="test",
        chunking=ChunkingConfig(),
        retrieval=RetrievalConfig(),
        sources=(),
    )


class FakePinecone:
    """Minimal stand-in for the Pinecone client used by the guards."""

    def __init__(self, exists: bool = True, dimension: int = 1536):
        self._exists = exists
        self._dimension = dimension
        self.created = None

    def has_index(self, name):
        return self._exists

    def create_index(self, name, dimension, metric, spec):
        self.created = {"name": name, "dimension": dimension, "metric": metric, "spec": spec}
        self._exists = True
        self._dimension = dimension

    def describe_index(self, name):
        return types.SimpleNamespace(dimension=self._dimension, status={"ready": True})

    def Index(self, name):  # noqa: N802 - mirrors the real client's method name
        return object()


@pytest.fixture(autouse=True)
def no_real_embeddings(monkeypatch):
    """_wrap_store builds an OpenAIEmbeddings client; stub it out."""
    monkeypatch.setattr(pinecone_store, "OpenAIEmbeddings", lambda **kwargs: object())
    monkeypatch.setattr(pinecone_store, "PineconeVectorStore", lambda **kwargs: kwargs)


# --- unknown model ---


def test_unknown_embedding_model_fails_loudly():
    with pytest.raises(RuntimeError, match="Unknown embedding model 'made-up-model'"):
        pinecone_store.embedding_dimension(make_cfg("made-up-model"))


def test_known_models_have_their_dimensions():
    assert pinecone_store.embedding_dimension(make_cfg("text-embedding-3-small")) == 1536
    assert pinecone_store.embedding_dimension(make_cfg("text-embedding-3-large")) == 3072


# --- dimension mismatch ---


def test_dimension_mismatch_is_rejected_at_ingestion():
    client = FakePinecone(exists=True, dimension=3072)
    with pytest.raises(RuntimeError, match="has dimension 3072 but"):
        pinecone_store.get_vector_store(make_cfg(), client=client)


def test_dimension_mismatch_is_rejected_at_query_time():
    client = FakePinecone(exists=True, dimension=768)
    with pytest.raises(RuntimeError, match="has dimension 768 but"):
        pinecone_store.get_vector_store_for_query(make_cfg(), client=client)


# --- missing index: create at ingestion, fail at query time ---


def test_missing_index_is_created_at_ingestion():
    client = FakePinecone(exists=False)
    pinecone_store.get_vector_store(make_cfg(), client=client)
    assert client.created["name"] == "test-index"
    assert client.created["dimension"] == 1536
    assert client.created["metric"] == "cosine"


def test_index_metric_comes_from_config():
    cfg = make_cfg()
    client = FakePinecone(exists=False)
    pinecone_store.get_vector_store(
        RagConfig(**{**cfg.__dict__, "pinecone_index_metric": "dotproduct"}), client=client
    )
    assert client.created["metric"] == "dotproduct"


def test_missing_index_fails_at_query_time_instead_of_being_created():
    client = FakePinecone(exists=False)
    with pytest.raises(RuntimeError, match="does not exist"):
        pinecone_store.get_vector_store_for_query(make_cfg(), client=client)
    assert client.created is None, "query path must never create an index"


# --- upsert ---


def test_upsert_passes_deterministic_ids_through():
    class FakeStore:
        def __init__(self):
            self.calls = []

        def add_documents(self, docs, ids):
            self.calls.append((docs, ids))

    doc = types.SimpleNamespace(id="src/file#0")
    store = FakeStore()
    pinecone_store.upsert_chunks(make_cfg(), [doc], store=store)
    assert store.calls == [([doc], ["src/file#0"])]

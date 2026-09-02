"""build_retriever: recursive dispatch over pipeline configs (design_summary.md 2.7)."""

import pytest
from rag_core.retriever.bm25 import BM25Retriever
from rag_core.retriever.dense import DenseRetriever
from rag_core.retriever.factory import build_retriever


class _FakeEmbeddings:
    """Deterministic, dimension-3 embeddings so cosine ranking is checkable."""

    _VECTORS = {
        "cats are great": [1.0, 0.0, 0.0],
        "dogs are great": [0.0, 1.0, 0.0],
    }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._VECTORS[t] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._VECTORS.get(text, [1.0, 0.0, 0.0])


class _Resources:
    def __init__(self, corpus, embeddings=None):
        self.corpus = corpus
        self._embeddings = embeddings

    def get_embeddings(self, name):
        return self._embeddings

    def get_dense_vectors(self, name):
        return None


_CORPUS = {"d1": "cats are great", "d2": "dogs are great"}


def test_build_retriever_bm25():
    cfg = {"type": "bm25", "k1": 0.9, "b": 0.4, "top_k": 10}
    retriever = build_retriever(cfg, _Resources(_CORPUS))
    assert isinstance(retriever, BM25Retriever)


def test_build_retriever_dense():
    cfg = {"type": "dense", "embeddings": "default", "metric": "cosine", "top_k": 10}
    retriever = build_retriever(cfg, _Resources(_CORPUS, embeddings=_FakeEmbeddings()))
    assert isinstance(retriever, DenseRetriever)

    results = retriever.search("cats are great", k=1)
    assert results[0].doc_id == "d1"


def test_build_retriever_unknown_type_raises():
    with pytest.raises(ValueError, match="unknown"):
        build_retriever({"type": "hyde"}, _Resources(_CORPUS))

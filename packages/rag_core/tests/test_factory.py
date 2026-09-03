"""build_retriever: recursive dispatch over pipeline configs (design_summary.md 2.7, 4.5)."""

import pytest
from rag_core.retriever.bm25 import BM25Retriever
from rag_core.retriever.dense import DenseRetriever
from rag_core.retriever.factory import build_retriever
from rag_core.retriever.fusion import RRFRetriever
from rag_core.retriever.rerank import BiEncoderScorer, RerankingRetriever


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
    def __init__(self, corpus, embeddings=None, reranker=None):
        self.corpus = corpus
        self._embeddings = embeddings
        self._reranker = reranker

    def get_embeddings(self, name):
        return self._embeddings

    def get_dense_vectors(self, name):
        return None

    def get_reranker(self, name):
        return self._reranker


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


def test_build_retriever_rrf_recurses_over_retrievers():
    cfg = {
        "type": "rrf",
        "rrf_k": 60,
        "top_k": 10,
        "retrievers": [
            {"type": "bm25", "k1": 0.9, "b": 0.4, "top_k": 10},
            {"type": "dense", "embeddings": "default", "metric": "cosine", "top_k": 10},
        ],
    }
    retriever = build_retriever(cfg, _Resources(_CORPUS, embeddings=_FakeEmbeddings()))
    assert isinstance(retriever, RRFRetriever)

    results = retriever.search("cats are great", k=2)
    assert {r.doc_id for r in results} == {"d1", "d2"}
    assert all(r.score_type == "rrf" for r in results)


def test_build_retriever_rerank_recurses_into_inner():
    cfg = {
        "type": "rerank",
        "reranker": "bi_encoder",
        "candidate_k": 2,
        "top_k": 1,
        "inner": {"type": "bm25", "k1": 0.9, "b": 0.4, "top_k": 10},
    }
    scorer = BiEncoderScorer(embeddings=_FakeEmbeddings())
    retriever = build_retriever(cfg, _Resources(_CORPUS, reranker=scorer))
    assert isinstance(retriever, RerankingRetriever)

    results = retriever.search("cats are great", k=1)
    assert len(results) == 1
    assert results[0].score_type == "rerank_logit"


def test_build_retriever_nested_hybrid_cross_encoder():
    """rerank -> rrf -> [bm25, dense], the full nested shape from design.md."""
    cfg = {
        "type": "rerank",
        "reranker": "cross_encoder",
        "candidate_k": 50,
        "top_k": 10,
        "inner": {
            "type": "rrf",
            "rrf_k": 60,
            "top_k": 10,
            "retrievers": [
                {"type": "bm25", "k1": 0.9, "b": 0.4, "top_k": 10},
                {"type": "dense", "embeddings": "default", "metric": "cosine", "top_k": 10},
            ],
        },
    }
    scorer = BiEncoderScorer(embeddings=_FakeEmbeddings())
    retriever = build_retriever(cfg, _Resources(_CORPUS, embeddings=_FakeEmbeddings(), reranker=scorer))

    assert isinstance(retriever, RerankingRetriever)
    assert isinstance(retriever._inner, RRFRetriever)
    assert all(isinstance(r, (BM25Retriever, DenseRetriever)) for r in retriever._inner._retrievers)


def test_build_retriever_nested_config_honors_depth_rule():
    """The inner RRF must receive candidate_k (50), not the 10 its children configure
    (design.md §7: a wrong top_k/candidate_k interaction degrades results silently).
    """
    calls: list[int | None] = []

    class _RecordingRRF:
        def search(self, query, k=None):
            calls.append(k)
            return []

    cfg = {
        "type": "rerank",
        "reranker": "bi_encoder",
        "candidate_k": 50,
        "top_k": 10,
        "inner": {"type": "bm25", "k1": 0.9, "b": 0.4, "top_k": 10},  # placeholder, replaced below
    }
    scorer = BiEncoderScorer(embeddings=_FakeEmbeddings())
    retriever = build_retriever(cfg, _Resources(_CORPUS, reranker=scorer))
    retriever._inner = _RecordingRRF()  # swap in a spy after construction

    retriever.search("cats are great", k=10)

    assert calls == [50]

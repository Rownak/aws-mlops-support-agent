"""build_pipeline_retriever: index cache reuse across composite pipelines (4.7)."""

from rag_bench_eval import embedding_cache, index_cache
from rag_bench_eval.resources import _embeddings_cache, _reranker_cache
from rag_bench_eval.retrievers import build_pipeline_retriever

_CORPUS = {"d1": "cats are great", "d2": "dogs are great", "d3": "cats and dogs"}


class _FakeEmbeddings:
    _VECTORS = {
        "cats are great": [1.0, 0.0],
        "dogs are great": [0.0, 1.0],
        "cats and dogs": [0.7, 0.7],
    }

    def __init__(self):
        self.embed_documents_calls = 0

    def embed_documents(self, texts):
        self.embed_documents_calls += 1
        return [self._VECTORS[t] for t in texts]

    def embed_query(self, text):
        return self._VECTORS[text]


def _config():
    return {
        "embeddings": {"default": {"provider": "fake", "model": "fake-model"}},
        "rerankers": {
            "bi_encoder": {"provider": "bi_encoder", "embeddings": "default"},
        },
        "retrieval": {
            "pipelines": {
                "dense": {
                    "type": "dense",
                    "embeddings": "default",
                    "metric": "cosine",
                    "top_k": 10,
                },
                "hybrid": {
                    "type": "rrf",
                    "rrf_k": 60,
                    "top_k": 10,
                    "retrievers": [
                        {"type": "bm25", "k1": 0.9, "b": 0.4, "top_k": 10},
                        {
                            "type": "dense",
                            "embeddings": "default",
                            "metric": "cosine",
                            "top_k": 10,
                        },
                    ],
                },
            }
        },
    }


def _reset_caches(tmp_path, monkeypatch):
    index_cache.clear()
    _embeddings_cache.clear()
    _reranker_cache.clear()
    monkeypatch.setattr(embedding_cache, "EMBEDDINGS_CACHE_DIR", tmp_path)
    monkeypatch.setattr("rag_bench_eval.retrievers.embedding_cache", embedding_cache)


def test_sweep_of_dense_and_hybrid_embeds_corpus_once(tmp_path, monkeypatch):
    _reset_caches(tmp_path, monkeypatch)
    fake_embeddings = _FakeEmbeddings()
    monkeypatch.setattr(
        "rag_bench_eval.resources.get_embedding", lambda cfg: fake_embeddings
    )

    config = _config()
    build_pipeline_retriever("dense", config["retrieval"]["pipelines"]["dense"], config, _CORPUS)
    build_pipeline_retriever("hybrid", config["retrieval"]["pipelines"]["hybrid"], config, _CORPUS)

    assert fake_embeddings.embed_documents_calls == 1


def test_hybrid_rrf_retriever_composes_bm25_and_dense_leaves(tmp_path, monkeypatch):
    _reset_caches(tmp_path, monkeypatch)
    fake_embeddings = _FakeEmbeddings()
    monkeypatch.setattr(
        "rag_bench_eval.resources.get_embedding", lambda cfg: fake_embeddings
    )

    config = _config()
    retriever = build_pipeline_retriever(
        "hybrid", config["retrieval"]["pipelines"]["hybrid"], config, _CORPUS
    )

    results = retriever.search("cats are great", k=2)
    assert {r.doc_id for r in results}.issubset(_CORPUS.keys())
    assert all(r.score_type == "rrf" for r in results)

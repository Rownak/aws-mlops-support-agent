"""Build a Retriever from one benchmark.yaml pipeline entry.

Wraps rag_core's build_retriever with this package's two runtime caches:
the disk-persisted corpus embedding matrix (2.6) and the in-process index
cache (2.5), keyed on the settings that actually change the index. Neither
cache is visible to rag_core — build_retriever itself stays a pure
dispatcher (design_summary.md 2.7); it only takes an optional pre-built
vectors matrix and hands back whatever it built, including a fresh one.
"""

import numpy as np
from rag_core.retriever.base import Retriever
from rag_core.retriever.dense import DenseRetriever
from rag_core.retriever.factory import build_retriever

from rag_bench_eval import embedding_cache
from rag_bench_eval.index_cache import get_or_build
from rag_bench_eval.resources import get_embeddings, get_embeddings_model_name


class _Resources:
    """Satisfies rag_core.retriever.factory.Resources for one build_retriever() call."""

    def __init__(self, corpus: dict[str, str], config: dict):
        self.corpus = corpus
        self._config = config
        self._cached_vectors: dict[str, np.ndarray | None] = {}

    def get_embeddings(self, name: str):
        return get_embeddings(name, self._config)

    def get_dense_vectors(self, name: str) -> np.ndarray | None:
        model_name = get_embeddings_model_name(name, self._config)
        cached = embedding_cache.load(model_name, self.corpus)
        if cached is None:
            self._cached_vectors[name] = None
            return None

        doc_ids, vectors = cached
        # The cache stores vectors aligned to the corpus's key order at save
        # time; DenseRetriever assumes the same order, so verify rather than
        # silently misalign scores to the wrong doc_ids.
        assert doc_ids == list(self.corpus.keys()), "cached doc_id order does not match corpus"
        self._cached_vectors[name] = vectors
        return vectors

    def had_cache_hit(self, name: str) -> bool:
        return self._cached_vectors.get(name) is not None


def build_pipeline_retriever(
    name: str, pipeline_cfg: dict, config: dict, corpus: dict[str, str]
) -> Retriever:
    ptype = pipeline_cfg["type"]

    if ptype == "dense":
        embeddings_name = pipeline_cfg["embeddings"]
        metric = pipeline_cfg["metric"]

        def build() -> DenseRetriever:
            resources = _Resources(corpus, config)
            retriever = build_retriever(pipeline_cfg, resources)
            if not resources.had_cache_hit(embeddings_name):
                model_name = get_embeddings_model_name(embeddings_name, config)
                embedding_cache.save(
                    model_name, corpus, list(corpus.keys()), np.asarray(retriever.vectors)
                )
            return retriever

        return get_or_build(("dense", embeddings_name, metric), build)

    if ptype == "bm25":
        k1, b = pipeline_cfg["k1"], pipeline_cfg["b"]
        return get_or_build(
            ("bm25", k1, b),
            lambda: build_retriever(pipeline_cfg, _Resources(corpus, config)),
        )

    raise ValueError(f"unknown pipeline type: {ptype!r} (pipeline {name!r})")

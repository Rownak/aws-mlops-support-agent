"""Build a Retriever from one benchmark.yaml pipeline entry.

Wraps rag_core's build_retriever with this package's two runtime caches:
the disk-persisted corpus embedding matrix (2.6) and the in-process index
cache (2.5), keyed on the settings that actually change the index. Neither
cache is visible to rag_core — build_retriever itself stays a pure
dispatcher (design_summary.md 2.7); it only takes an optional pre-built
vectors matrix and hands back whatever it built, including a fresh one.

`rrf` and `rerank` nodes recurse through this module's own dispatch (4.7),
not rag_core's — a composite's leaves must hit the cache above just like a
top-level bm25/dense pipeline does, so `hybrid`'s bm25 and dense children
reuse whatever `bm25`/`dense` already built in the same sweep rather than
rebuilding. The composite wrapper itself (RRFRetriever/RerankingRetriever)
is cheap to construct and isn't cached — only its embedding/BM25-index
leaves are.
"""

import numpy as np
from rag_core.retriever.base import Retriever
from rag_core.retriever.dense import DenseRetriever
from rag_core.retriever.factory import build_retriever
from rag_core.retriever.fusion import RRFRetriever
from rag_core.retriever.rerank import RerankingRetriever

from rag_bench_eval import embedding_cache
from rag_bench_eval.index_cache import get_or_build
from rag_bench_eval.resources import get_embeddings, get_embeddings_model_name
from rag_bench_eval.resources import get_reranker as get_reranker_resource


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

    def get_reranker(self, name: str):
        return get_reranker_resource(name, self._config)


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

    if ptype == "rrf":
        children = [
            build_pipeline_retriever(name, child_cfg, config, corpus)
            for child_cfg in pipeline_cfg["retrievers"]
        ]
        return RRFRetriever(
            retrievers=children,
            rrf_k=pipeline_cfg["rrf_k"],
            top_k=pipeline_cfg["top_k"],
        )

    if ptype == "rerank":
        inner = build_pipeline_retriever(name, pipeline_cfg["inner"], config, corpus)
        return RerankingRetriever(
            inner=inner,
            scorer=get_reranker_resource(pipeline_cfg["reranker"], config),
            candidate_k=pipeline_cfg["candidate_k"],
            top_k=pipeline_cfg["top_k"],
        )

    raise ValueError(f"unknown pipeline type: {ptype!r} (pipeline {name!r})")

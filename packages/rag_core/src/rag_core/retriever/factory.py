"""Build a Retriever from a benchmark.yaml pipeline config (design_summary.md).

Recursive `if cfg["type"] == ...` dispatch — no schema layer, no registry.
An unknown `type` raises at load time, naming the type and the valid set.

`resources` is a small object the caller controls: it must expose
`corpus: dict[str, str]` and `get_embeddings(name: str) -> Embeddings`.
Keeping the interface this narrow lets rag_bench_eval own its own resource
caching (embeddings dict cache, index cache) without rag_core knowing about
either.

`get_dense_vectors` is optional: a caller with a disk cache of the corpus
matrix (design_summary.md 2.6) can return it pre-built to skip re-embedding;
returning None (the default) falls through to a normal embed.
"""

from typing import Any, Protocol

from rag_core.retriever.base import Retriever
from rag_core.retriever.bm25 import BM25Retriever
from rag_core.retriever.dense import DenseRetriever

_VALID_TYPES = ("bm25", "dense")


class Resources(Protocol):
    corpus: dict[str, str]

    def get_embeddings(self, name: str) -> Any: ...

    def get_dense_vectors(self, name: str) -> Any | None: ...


def build_retriever(cfg: dict, resources: Resources) -> Retriever:
    ptype = cfg["type"]

    if ptype == "bm25":
        return BM25Retriever(
            corpus=resources.corpus,
            k1=cfg["k1"],
            b=cfg["b"],
            top_k=cfg["top_k"],
        )

    if ptype == "dense":
        embeddings_name = cfg["embeddings"]
        return DenseRetriever(
            corpus=resources.corpus,
            embeddings=resources.get_embeddings(embeddings_name),
            top_k=cfg["top_k"],
            vectors=resources.get_dense_vectors(embeddings_name),
        )

    raise ValueError(f"unknown pipeline type: {ptype!r}. Valid types: {_VALID_TYPES}")

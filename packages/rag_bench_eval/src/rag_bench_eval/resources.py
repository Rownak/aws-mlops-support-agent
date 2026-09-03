"""Named resources from benchmark.yaml's `embeddings:`/`rerankers:` maps
(design_summary.md).

Dict caches over rag_core's provider dispatch, so `embeddings: default` or
`reranker: bi_encoder` referenced from several pipelines in one sweep builds
once. Bare embeddings names only for now — `{query:, passage:}` asymmetric
encoders are designed-for but not implemented (design.md §8 Q4).
"""

from typing import Any

from rag_core.embeddings.factory import get_embedding
from rag_core.retriever.rerank import BiEncoderScorer, RelevanceScorer, get_reranker as _get_scorer

_embeddings_cache: dict[str, Any] = {}
_embeddings_model_names: dict[str, str] = {}
_reranker_cache: dict[str, RelevanceScorer] = {}


def get_embeddings(name: str, cfg: dict) -> Any:
    """Look up `name` in `cfg["embeddings"]`, caching the built instance."""
    if name in _embeddings_cache:
        return _embeddings_cache[name]

    embeddings_cfg = cfg["embeddings"][name]
    instance = get_embedding(embeddings_cfg)
    _embeddings_cache[name] = instance
    _embeddings_model_names[name] = embeddings_cfg["model"]
    return instance


def get_embeddings_model_name(name: str, cfg: dict) -> str:
    """The model string behind resource `name` — the disk embedding cache's key."""
    get_embeddings(name, cfg)  # ensure it's resolved at least once
    return _embeddings_model_names[name]


def get_reranker(name: str, cfg: dict) -> RelevanceScorer:
    """Look up `name` in `cfg["rerankers"]`, caching the built instance.

    `provider` is the discriminator here (not `type`, which in `pipelines:`
    means topology — design.md §3.3). `bi_encoder` needs a pre-built
    Embeddings instance, which rag_core.retriever.rerank.get_reranker()
    cannot resolve on its own (it only splats primitive config values into
    a constructor) — so that provider is resolved here, and every other
    provider (cross_encoder, cohere) still goes through rag_core's factory.
    """
    if name in _reranker_cache:
        return _reranker_cache[name]

    reranker_cfg = cfg["rerankers"][name]
    provider = reranker_cfg["provider"]

    if provider == "bi_encoder":
        embeddings = get_embeddings(reranker_cfg["embeddings"], cfg)
        instance = BiEncoderScorer(embeddings=embeddings)
    else:
        instance = _get_scorer(reranker_cfg)

    _reranker_cache[name] = instance
    return instance

"""Named resources from benchmark.yaml's `embeddings:` map (design_summary.md).

A dict cache over rag_core's provider dispatch, so `embeddings: default`
referenced from two pipelines in one sweep loads the model once. Bare names
only for now — `{query:, passage:}` asymmetric encoders are designed-for but
not implemented (design.md §8 Q4).
"""

from typing import Any

from rag_core.embeddings.factory import get_embedding

_embeddings_cache: dict[str, Any] = {}
_embeddings_model_names: dict[str, str] = {}


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

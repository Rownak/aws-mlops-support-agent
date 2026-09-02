"""Disk cache for dense corpus embeddings (design_summary.md 2.6).

Keyed by model name + corpus hash, so a corpus edit invalidates stale
vectors instead of silently reusing them. Vectors go in a `.npy` file;
doc_ids go in a `.json` sidecar, since `.npy` alone can't carry them and a
cache hit must still know which row belongs to which doc.

Lives in rag_bench_eval, not rag_core: the path convention
(`data/beir/nfcorpus/.embeddings/`) is this benchmark harness's, not a
generic retriever concern (DenseRetriever itself has no disk I/O).
"""

import hashlib
import json

import numpy as np

from rag_bench_eval.settings import EMBEDDINGS_CACHE_DIR


def corpus_hash(corpus: dict[str, str]) -> str:
    """Stable hash over doc_id + text pairs, order-independent."""
    h = hashlib.sha256()
    for doc_id in sorted(corpus):
        h.update(doc_id.encode("utf-8"))
        h.update(b"\0")
        h.update(corpus[doc_id].encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _paths(model: str, corpus_digest: str) -> tuple:
    key = f"{model}__{corpus_digest}"
    return EMBEDDINGS_CACHE_DIR / f"{key}.npy", EMBEDDINGS_CACHE_DIR / f"{key}.json"


def load(model: str, corpus: dict[str, str]) -> tuple[list[str], "np.ndarray"] | None:
    """Return (doc_ids, vectors) aligned to each other, or None on a cache miss."""
    vectors_path, doc_ids_path = _paths(model, corpus_hash(corpus))
    if not (vectors_path.exists() and doc_ids_path.exists()):
        return None

    doc_ids = json.loads(doc_ids_path.read_text(encoding="utf-8"))
    vectors = np.load(vectors_path)
    return doc_ids, vectors


def save(model: str, corpus: dict[str, str], doc_ids: list[str], vectors: "np.ndarray") -> None:
    EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vectors_path, doc_ids_path = _paths(model, corpus_hash(corpus))
    np.save(vectors_path, vectors)
    doc_ids_path.write_text(json.dumps(doc_ids), encoding="utf-8")

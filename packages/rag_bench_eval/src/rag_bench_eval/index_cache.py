"""Cache built retriever indexes on their settings (design_summary.md).

Embedding 3,633 docs is the one slow step in a sweep — keyed on the settings
that actually change the index (embeddings name + metric for dense; k1/b for
BM25), so an identical pipeline referenced twice in one run embeds once.
Runtime only: this cache never appears in benchmark.yaml and does not
persist across processes (that's 2.6, disk-cached vectors).
"""

from typing import Any, Callable

_index_cache: dict[tuple, Any] = {}


def get_or_build(key: tuple, build: Callable[[], Any]) -> Any:
    if key not in _index_cache:
        _index_cache[key] = build()
    return _index_cache[key]


def clear() -> None:
    """Mainly for tests: reset the cache between runs that reuse the same key."""
    _index_cache.clear()

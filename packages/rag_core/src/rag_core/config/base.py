"""Shared behaviour for the config dataclasses.

``DictLike`` exists because four of the seven leaf configs must render
themselves as a plain dict: the subpackages they feed
(``embeddings.factory.get_embedding``, ``llm.factory.get_llm``,
``vectorstores.pinecone_store.PineconeStore``, ``sources.build_sources``)
all take a dict, deliberately, so nothing downstream of config parsing is
coupled to these types.

The other three (``SplitterConfig``, ``RetrieverConfig``,
``GenerationConfig``) intentionally do NOT get ``as_dict()``. Their consumers
take scalar keyword arguments instead (see ``pipeline.py``: ``_splitter()``
reads ``.chunk_size``/``.chunk_overlap``, and ``assess_confidence`` takes
``min_top_score=``). Giving them an unused ``as_dict()`` would imply a
dict-driven contract that does not exist.
"""

from dataclasses import fields


class DictLike:
    """Render a frozen dataclass as the plain dict its consumer expects.

    Subclasses set ``_optional`` to the field names that should be dropped
    when they are None. Everything else is emitted unconditionally — which
    matters: ``api_key`` stays in the dict even when None (the factories call
    ``config.get("api_key")`` and pass it straight through), while ``host``
    and ``num_ctx`` must vanish so the constructors fall back to their own
    defaults.
    """

    #: Field names omitted from the dict when their value is None.
    _optional: frozenset = frozenset()

    #: Field names to mask when rendering for humans. Read by `describe()`.
    _secret_fields: frozenset = frozenset()

    def as_dict(self) -> dict:
        # asdict() would deep-copy nested dataclasses; these leaves hold only
        # scalars, so the shallow field walk is both correct and cheaper.
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in self._optional or getattr(self, f.name) is not None
        }

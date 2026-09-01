"""Tests for processing.chunking.build_chunks.

Chunking is a free function with no vector store and no config file, so
these need no RagCore, no fakes beyond a splitter, and no network.
"""

import pytest
from rag_core.processing.chunking import RESERVED_METADATA_KEYS, build_chunks


class _FakeSplitter:
    """Splits on blank lines, so chunk counts are predictable in assertions."""

    def split_text(self, text):
        return [p for p in text.split("\n\n") if p.strip()]


def _build(text="one\n\ntwo", **kwargs):
    defaults = dict(
        file_path="docs/a.md",
        file_name="a.md",
        file_type="md",
        file_hash="abc123",
        splitter=_FakeSplitter(),
    )
    return build_chunks(text=text, **{**defaults, **kwargs})


def test_build_chunks_stamps_provenance_metadata():
    chunks = _build()

    assert len(chunks) == 2
    assert [c.metadata["chunk_index"] for c in chunks] == [0, 1]
    assert all(c.metadata["total_chunks"] == 2 for c in chunks)
    assert all(c.metadata["file_hash"] == "abc123" for c in chunks)
    # Identical text in two chunks would still hash differently, because
    # chunk_hash mixes in chunk_id.
    assert chunks[0].metadata["chunk_hash"] != chunks[1].metadata["chunk_hash"]


def test_build_chunks_without_extra_metadata_adds_no_keys():
    """The no-extra-metadata path must stay exactly what it always was."""
    chunks = _build()

    assert set(chunks[0].metadata) == RESERVED_METADATA_KEYS


def test_build_chunks_merges_extra_metadata():
    chunks = _build(extra_metadata={"url": "https://docs.example.com/a.html"})

    assert len(chunks) == 2
    # Every chunk of the file carries it, and the reserved keys survive.
    for chunk in chunks:
        assert chunk.metadata["url"] == "https://docs.example.com/a.html"
        assert RESERVED_METADATA_KEYS <= set(chunk.metadata)


@pytest.mark.parametrize("key", ["file_hash", "source", "chunk_index"])
def test_build_chunks_rejects_reserved_key_collision(key):
    """Overwriting a reserved key would silently break dedup / partial-ingest
    detection, so it is refused rather than merged."""
    with pytest.raises(ValueError, match="reserved key"):
        _build(extra_metadata={key: "clobbered"})


def test_build_chunks_reserved_collision_names_the_offending_key():
    with pytest.raises(ValueError, match="file_hash"):
        _build(extra_metadata={"file_hash": "x", "url": "fine"})


def test_build_chunks_empty_text_yields_nothing():
    assert _build(text="   ") == []

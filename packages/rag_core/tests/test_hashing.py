"""Tests for content-hashing helpers used for ingest deduplication."""

import pytest

from rag_core.processing.hashing import (
    compare_hashes,
    sha256_chunk,
    sha256_file,
    sha256_file_from_path,
    sha256_text,
)


def test_sha256_text_is_deterministic():
    assert sha256_text("hello") == sha256_text("hello")


def test_sha256_text_differs_for_different_content():
    assert sha256_text("hello") != sha256_text("world")


def test_sha256_file_matches_text_hash_for_utf8_content():
    text = "hello"
    assert sha256_file(text.encode("utf-8")) == sha256_text(text)


def test_sha256_file_from_path(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_bytes(b"some content")
    assert sha256_file_from_path(path) == sha256_file(b"some content")


def test_sha256_file_from_path_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        sha256_file_from_path(tmp_path / "nope.txt")


def test_sha256_chunk_incorporates_chunk_id():
    a = sha256_chunk("chunk-1", "same text")
    b = sha256_chunk("chunk-2", "same text")
    assert a != b


def test_compare_hashes():
    h = sha256_text("x")
    assert compare_hashes(h, h)
    assert not compare_hashes(h, sha256_text("y"))

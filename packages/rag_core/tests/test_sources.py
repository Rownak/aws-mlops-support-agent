"""Tests for the local-directory document source."""

import pytest

from rag_core.sources.base import build_source, build_sources
from rag_core.sources.local import LocalSource


def test_missing_path_raises(tmp_path):
    source = LocalSource(path=str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError):
        source.list_files()


def test_lists_matching_files_sorted(tmp_path):
    (tmp_path / "b.md").write_text("b")
    (tmp_path / "a.md").write_text("a")
    (tmp_path / "skip.png").write_text("x")

    source = LocalSource(path=str(tmp_path), extensions=[".md"])
    files = source.list_files()

    assert [f.split("\\")[-1].split("/")[-1] for f in files] == ["a.md", "b.md"]


def test_recursive_flag_controls_descent(tmp_path):
    (tmp_path / "top.md").write_text("t")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "deep.md").write_text("d")

    shallow = LocalSource(path=str(tmp_path), recursive=False, extensions=[".md"])
    assert len(shallow.list_files()) == 1

    deep = LocalSource(path=str(tmp_path), recursive=True, extensions=[".md"])
    assert len(deep.list_files()) == 2


def test_build_source_from_config(tmp_path):
    source = build_source({"type": "local", "path": str(tmp_path)})
    assert isinstance(source, LocalSource)


def test_build_source_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown source type"):
        build_source({"type": "nonexistent"})


def test_build_source_requires_type_key():
    with pytest.raises(ValueError, match="missing a 'type' key"):
        build_source({"path": "./docs"})


def test_build_sources_returns_empty_list_for_none():
    assert build_sources(None) == []


def test_build_sources_builds_every_entry(tmp_path):
    sources = build_sources([{"type": "local", "path": str(tmp_path)}])
    assert len(sources) == 1
    assert isinstance(sources[0], LocalSource)

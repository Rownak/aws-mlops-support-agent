"""Tests for RagCore.ingest_documents / ingest_directory / sync.

RagCore.__init__ builds real provider clients, so these bypass it with
__new__ and inject fakes, exercising the prepare -> write path with no
network and no real Pinecone.
"""

from pathlib import Path

import pytest
from rag_core.config import SplitterConfig, VectorStoreConfig
from rag_core.ingestion import Ingestor
from rag_core.pipeline import RagCore
from rag_core.processing.chunking import RESERVED_METADATA_KEYS


class _FakeSplitter:
    """Splits on blank lines, so chunk counts are predictable in assertions."""

    def split_text(self, text):
        return [p for p in text.split("\n\n") if p.strip()]


class _FakeLoader:
    def __init__(self, results=None, fail_for=None):
        self.results = results or {}
        self.fail_for = fail_for or set()

    def load(self, path):
        if path in self.fail_for:
            return {"success": False, "error": "boom", "text_content": "",
                    "file_name": path, "file_type": "md"}
        return {
            "success": True,
            "error": None,
            "text_content": self.results.get(path, "chunk one\n\nchunk two"),
            # Matches MarkItDownLoader, which uses Path(...).name.
            "file_name": Path(path).name,
            "file_type": "md",
        }


class _FakeVectorStore:
    def __init__(self, fail_on_write=False):
        self.added = []
        self.batches = []
        self.fail_on_write = fail_on_write

    def add_documents(self, docs):
        if self.fail_on_write:
            raise RuntimeError("write failed")
        self.batches.append(len(docs))
        self.added.extend(docs)


class _FakeStore:
    """Stands in for PineconeStore, recording the mutations it is asked for."""

    def __init__(self, vectorstore, status=("absent", 0, None), stale=0):
        self._vectorstore = vectorstore
        self._status = status
        self._stale = stale
        self.deleted_hashes = []
        self.deleted_sources = []
        self.created = False

    def create_collection(self, use_sparse=False):
        self.created = True

    def get_store(self, use_sparse=False):
        return self._vectorstore

    def get_ingest_status(self, file_hash):
        return self._status

    def delete_by_file_hash(self, file_hash):
        self.deleted_hashes.append(file_hash)
        return 1

    def delete_by_source(self, source, except_file_hash=None):
        self.deleted_sources.append((source, except_file_hash))
        return self._stale


def _rag_core(store, loader, batch_size=100, sources=()):
    rag = RagCore.__new__(RagCore)
    rag.config = type(
        "Cfg",
        (),
        {
            "vectorstore": VectorStoreConfig(),
            "splitter": SplitterConfig(),
            "sources": sources,
            "loader_extensions": (".md", ".txt"),
        },
    )()
    rag.store = store
    rag.loader = loader
    # The ingest verbs delegate here, so the fakes have to reach the
    # Ingestor rather than only the facade.
    rag.ingestor = Ingestor(rag.config, store, loader, batch_size=batch_size)
    return rag


@pytest.fixture(autouse=True)
def _fake_splitter(monkeypatch):
    """Chunking now lives in processing.chunking, so the splitter is injected
    by patching what ingest_documents calls rather than a RagCore method."""
    monkeypatch.setattr(
        "rag_core.ingestion.ingestor.splitter_from_config", lambda cfg: _FakeSplitter()
    )


@pytest.fixture
def docs(tmp_path):
    """Two real files on disk, so hashing works against actual bytes."""
    a = tmp_path / "a.md"
    a.write_text("chunk one\n\nchunk two", encoding="utf-8")
    b = tmp_path / "b.md"
    b.write_text("only one", encoding="utf-8")
    return str(a), str(b)


# --- the happy path ---


def test_ingest_documents_writes_chunks_and_counts_them(docs):
    a, b = docs
    vs = _FakeVectorStore()
    rag = _rag_core(_FakeStore(vs), _FakeLoader())

    stats = rag.ingest_documents([a, b])

    assert stats["total"] == 2
    assert stats["processed"] == 2
    assert stats["failed"] == 0
    assert stats["chunks_created"] == len(vs.added)
    assert stats["errors"] == []


def test_ingest_documents_stamps_provenance_metadata(docs):
    a, _ = docs
    vs = _FakeVectorStore()
    rag = _rag_core(_FakeStore(vs), _FakeLoader())

    rag.ingest_documents([a])

    meta = vs.added[0].metadata
    assert meta["source"] == a
    assert meta["file_name"] == "a.md"
    assert meta["chunk_index"] == 0
    # total_chunks is what makes a partial ingest detectable later.
    assert meta["total_chunks"] == len(vs.added)
    assert meta["file_hash"] and meta["chunk_hash"] and meta["content_hash"]
    # Nothing extra when no extra_metadata was passed.
    assert set(meta) == RESERVED_METADATA_KEYS


def test_ingest_documents_applies_extra_metadata_by_path(docs):
    """extra_metadata is per-path: only the named file's chunks get it."""
    a, b = docs
    vs = _FakeVectorStore()
    rag = _rag_core(_FakeStore(vs), _FakeLoader())

    rag.ingest_documents([a, b], extra_metadata={a: {"url": "https://x/a"}})

    for doc in vs.added:
        if doc.metadata["source"] == a:
            assert doc.metadata["url"] == "https://x/a"
        else:
            assert "url" not in doc.metadata


def test_ingest_documents_reserved_key_collision_fails_only_that_file(docs):
    """A source's bad metadata is one file's error, not a dead run."""
    a, b = docs
    vs = _FakeVectorStore()
    rag = _rag_core(_FakeStore(vs), _FakeLoader())

    stats = rag.ingest_documents([a, b], extra_metadata={a: {"chunk_index": 99}})

    assert stats["failed"] == 1
    assert stats["processed"] == 1
    assert stats["errors"][0]["file"] == a
    assert "reserved key" in stats["errors"][0]["error"]
    # b still landed, and with its own chunk_index intact.
    assert vs.added and all(d.metadata["source"] == b for d in vs.added)


def test_empty_file_list_returns_zeroed_stats():
    rag = _rag_core(_FakeStore(_FakeVectorStore()), _FakeLoader())
    stats = rag.ingest_documents([])
    assert stats["total"] == 0 and stats["processed"] == 0


# --- dedup / partial / replacement ---


def test_already_ingested_file_is_skipped(docs):
    a, _ = docs
    vs = _FakeVectorStore()
    store = _FakeStore(vs, status=("complete", 2, 2))
    rag = _rag_core(store, _FakeLoader())

    stats = rag.ingest_documents([a])

    assert stats["skipped"] == 1
    assert stats["processed"] == 0
    assert vs.added == []  # nothing re-embedded


def test_partial_ingest_is_cleared_then_rewritten(docs):
    a, _ = docs
    vs = _FakeVectorStore()
    store = _FakeStore(vs, status=("partial", 1, 2))
    rag = _rag_core(store, _FakeLoader())

    stats = rag.ingest_documents([a])

    assert store.deleted_hashes, "the partial ingest must be purged first"
    assert stats["processed"] == 1
    assert vs.added


def test_changed_file_replaces_its_previous_version(docs):
    a, _ = docs
    vs = _FakeVectorStore()
    store = _FakeStore(vs, stale=3)  # 3 stale chunks under a different hash
    rag = _rag_core(store, _FakeLoader())

    stats = rag.ingest_documents([a])

    source, except_hash = store.deleted_sources[0]
    assert source == a
    # The new version's hash is spared; only older ones are removed.
    assert except_hash == vs.added[0].metadata["file_hash"]
    assert stats["replaced"] == 1


# --- failure handling ---


def test_loader_failure_is_recorded_not_raised(docs):
    a, _ = docs
    rag = _rag_core(_FakeStore(_FakeVectorStore()), _FakeLoader(fail_for={a}))

    stats = rag.ingest_documents([a])

    assert stats["failed"] == 1
    assert stats["errors"][0]["file"] == a
    assert "boom" in stats["errors"][0]["error"]


def test_missing_file_is_recorded_not_raised(tmp_path):
    rag = _rag_core(_FakeStore(_FakeVectorStore()), _FakeLoader())

    stats = rag.ingest_documents([str(tmp_path / "nope.md")])

    assert stats["failed"] == 1
    assert stats["processed"] == 0


def test_document_with_no_extractable_text_is_a_failure_not_a_silent_pass(tmp_path):
    empty = tmp_path / "scanned.md"
    empty.write_text("   ", encoding="utf-8")
    rag = _rag_core(
        _FakeStore(_FakeVectorStore()), _FakeLoader(results={str(empty): "   "})
    )

    stats = rag.ingest_documents([str(empty)])

    # Counted in neither processed nor skipped, so it cannot go unnoticed.
    assert stats["failed"] == 1
    assert "no extractable text" in stats["errors"][0]["error"]


def test_write_failure_rolls_back_so_the_file_retries_cleanly(docs):
    a, _ = docs
    store = _FakeStore(_FakeVectorStore(fail_on_write=True))
    rag = _rag_core(store, _FakeLoader())

    stats = rag.ingest_documents([a])

    assert stats["failed"] == 1
    # Without the rollback the half-written chunks would look like a finished
    # document on the next run and be skipped forever.
    assert store.deleted_hashes


def test_one_bad_file_does_not_abort_the_run(docs):
    a, b = docs
    vs = _FakeVectorStore()
    rag = _rag_core(_FakeStore(vs), _FakeLoader(fail_for={a}))

    stats = rag.ingest_documents([a, b])

    assert stats["failed"] == 1 and stats["processed"] == 1


# --- batching ---


def test_chunks_are_written_in_batches(tmp_path):
    big = tmp_path / "big.md"
    big.write_text("x", encoding="utf-8")
    text = "\n\n".join(f"para {i}" for i in range(10))
    vs = _FakeVectorStore()
    rag = _rag_core(_FakeStore(vs), _FakeLoader(results={str(big): text}), batch_size=3)

    rag.ingest_documents([str(big)])

    assert vs.batches == [3, 3, 3, 1]
    assert len(vs.added) == 10


# --- ingest_directory ---


def test_ingest_directory_picks_up_matching_files(tmp_path):
    (tmp_path / "a.md").write_text("one\n\ntwo", encoding="utf-8")
    (tmp_path / "b.txt").write_text("three", encoding="utf-8")
    (tmp_path / "skip.png").write_text("nope", encoding="utf-8")
    rag = _rag_core(_FakeStore(_FakeVectorStore()), _FakeLoader())

    stats = rag.ingest_directory(str(tmp_path))

    assert stats["total"] == 2  # .png filtered out


def test_ingest_directory_recursive_flag(tmp_path):
    (tmp_path / "top.md").write_text("a", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "deep.md").write_text("b", encoding="utf-8")
    rag = _rag_core(_FakeStore(_FakeVectorStore()), _FakeLoader())

    assert rag.ingest_directory(str(tmp_path), recursive=False)["total"] == 1
    assert rag.ingest_directory(str(tmp_path), recursive=True)["total"] == 2


def test_ingest_directory_extensions_override(tmp_path):
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    rag = _rag_core(_FakeStore(_FakeVectorStore()), _FakeLoader())

    assert rag.ingest_directory(str(tmp_path), extensions=[".md"])["total"] == 1


def test_ingest_directory_rejects_a_non_directory(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("a", encoding="utf-8")
    rag = _rag_core(_FakeStore(_FakeVectorStore()), _FakeLoader())

    with pytest.raises(ValueError, match="Not a directory"):
        rag.ingest_directory(str(f))


def test_ingest_directory_with_no_matches_returns_zeroed_stats(tmp_path):
    (tmp_path / "skip.png").write_text("nope", encoding="utf-8")
    rag = _rag_core(_FakeStore(_FakeVectorStore()), _FakeLoader())

    stats = rag.ingest_directory(str(tmp_path))

    assert stats["total"] == 0 and stats["processed"] == 0


# --- sync shares the same path ---


def test_sync_ingests_what_the_sources_list(tmp_path, monkeypatch):
    a = tmp_path / "a.md"
    a.write_text("one\n\ntwo", encoding="utf-8")

    class _FakeSource:
        def list_files(self):
            return [str(a)]

    monkeypatch.setattr("rag_core.ingestion.ingestor.build_sources", lambda specs: [_FakeSource()])
    vs = _FakeVectorStore()
    rag = _rag_core(_FakeStore(vs), _FakeLoader())

    stats = rag.sync()

    # Same IngestStats shape as ingest_documents — one shared code path.
    assert stats["total"] == 1 and stats["processed"] == 1
    assert vs.added
    # This source predates metadata_for and never defines it; sync must not
    # require it, and must add nothing to the chunks.
    assert set(vs.added[0].metadata) == RESERVED_METADATA_KEYS


def test_sync_applies_each_sources_metadata_for(tmp_path, monkeypatch):
    a = tmp_path / "a.md"
    a.write_text("one\n\ntwo", encoding="utf-8")
    b = tmp_path / "b.md"
    b.write_text("only one", encoding="utf-8")

    class _UrlSource:
        def __init__(self, path, url):
            self.path, self.url = str(path), url

        def list_files(self):
            return [self.path]

        def metadata_for(self, file_path):
            return {"url": self.url}

    monkeypatch.setattr(
        "rag_core.ingestion.ingestor.build_sources",
        lambda specs: [_UrlSource(a, "https://x/a"), _UrlSource(b, "https://x/b")],
    )
    vs = _FakeVectorStore()
    rag = _rag_core(_FakeStore(vs), _FakeLoader())

    rag.sync()

    # Each source's metadata follows its own files, not the other's.
    by_url = {d.metadata["source"]: d.metadata["url"] for d in vs.added}
    assert by_url == {str(a): "https://x/a", str(b): "https://x/b"}

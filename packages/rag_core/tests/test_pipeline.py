"""End-to-end ingest test with a fake source and a fake store. No network.

This is the test that proves the pipeline is genuinely corpus-agnostic: the
source below is 20 lines of in-memory dict, and nothing in rag_core needed to
know anything about it.
"""

from rag_core.config import ChunkingConfig, RagConfig, RetrievalConfig, SourceSpec
from rag_core.pipeline import run_ingest
from rag_core.sources import LoadedDoc

CFG = RagConfig(
    openai_api_key="k",
    openai_chat_model="gpt-4o-mini",
    openai_embedding_model="text-embedding-3-small",
    pinecone_api_key="k",
    pinecone_index_name="test-index",
    pinecone_index_metric="cosine",
    aws_region="us-east-1",
    project="test-project",
    chunking=ChunkingConfig(size_tokens=800, overlap_tokens=100),
    retrieval=RetrievalConfig(),
    sources=(),
)


class FakeSource:
    """An in-memory DocSource — the whole corpus is a dict of markdown."""

    def __init__(self, source_id: str, docs: dict[str, str]):
        self.spec = SourceSpec(id=source_id, loader="fake")
        self._docs = docs

    def fetch(self):
        for name, text in self._docs.items():
            yield LoadedDoc(
                source_id=self.spec.id,
                source_file=name,
                text=text,
                url=f"https://example.invalid/{name.removesuffix('.md')}.html",
            )


class FakeUpsert:
    """Captures what the pipeline tried to write instead of calling Pinecone."""

    def __init__(self):
        self.docs = None
        self.calls = 0

    def __call__(self, cfg, docs, store=None):
        self.calls += 1
        self.docs = docs


def test_end_to_end_ingest_with_fakes():
    sources = [
        FakeSource("alpha", {"one.md": "# One\n\nalpha text", "two.md": "# Two\n\nmore text"}),
        FakeSource("beta", {"three.md": "# Three\n\nbeta text"}),
    ]
    upsert = FakeUpsert()

    report = run_ingest(CFG, sources, upsert=upsert)

    # Every source was processed and reported.
    assert report.chunks_per_source == {"alpha": 2, "beta": 1}
    assert report.total_chunks == 3
    assert report.index_name == "test-index"
    # One upsert for the whole run, carrying every chunk.
    assert upsert.calls == 1
    assert len(upsert.docs) == 3


def test_chunks_reaching_the_store_carry_full_provenance():
    upsert = FakeUpsert()
    run_ingest(CFG, [FakeSource("alpha", {"one.md": "# One\n\ntext"})], upsert=upsert)
    (doc,) = upsert.docs
    assert doc.id == "alpha/one#0"
    assert doc.metadata["source_id"] == "alpha"
    assert doc.metadata["source_file"] == "one.md"
    assert doc.metadata["heading"] == "One"
    assert doc.metadata["url"] == "https://example.invalid/one.html"


def test_ingest_is_deterministic_across_runs():
    """Same corpus in, same chunk IDs out — that is what makes re-runs idempotent."""
    source_docs = {"one.md": "# One\n\ntext"}
    first, second = FakeUpsert(), FakeUpsert()
    run_ingest(CFG, [FakeSource("alpha", source_docs)], upsert=first)
    run_ingest(CFG, [FakeSource("alpha", source_docs)], upsert=second)
    assert [d.id for d in first.docs] == [d.id for d in second.docs]


def test_empty_corpus_still_reports_cleanly():
    upsert = FakeUpsert()
    report = run_ingest(CFG, [FakeSource("alpha", {})], upsert=upsert)
    assert report.total_chunks == 0
    assert report.chunks_per_source == {"alpha": 0}
    assert upsert.docs == []


def test_report_summary_mentions_each_source_and_the_index():
    report = run_ingest(
        CFG, [FakeSource("alpha", {"one.md": "# One\n\ntext"})], upsert=FakeUpsert()
    )
    summary = report.summary()
    assert "alpha: 1" in summary
    assert "test-index" in summary


def test_chunking_config_is_honored_by_the_pipeline():
    """A tiny chunk budget must produce more chunks for the same document."""
    docs = {"big.md": "# Big\n\n" + ("word " * 2000)}
    small_cfg = RagConfig(**{**CFG.__dict__, "chunking": ChunkingConfig(size_tokens=100)})
    big, small = FakeUpsert(), FakeUpsert()
    run_ingest(CFG, [FakeSource("a", docs)], upsert=big)
    run_ingest(small_cfg, [FakeSource("a", docs)], upsert=small)
    assert len(small.docs) > len(big.docs)

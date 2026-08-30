"""Ingest orchestration: source -> load -> chunk -> embed -> upsert.

The whole pipeline is driven by config plus injected sources, so it never
learns what corpus it is processing. That is what makes it reusable, and what
makes it testable: `run_ingest` accepts a fake store and fake sources, so the
end-to-end path can be exercised with no network at all.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from langchain_core.documents import Document

from rag_core.chunking.markdown import chunk_documents
from rag_core.config import RagConfig
from rag_core.observability import log_event
from rag_core.sources import DocSource
from rag_core.vectorstore.pinecone_store import upsert_chunks


@dataclass(frozen=True)
class IngestReport:
    """What one ingestion run did — returned so callers can print or assert."""

    # chunks produced per source id, in the order the sources ran
    chunks_per_source: dict[str, int]
    total_chunks: int
    index_name: str

    def summary(self) -> str:
        per_source = ", ".join(f"{sid}: {n}" for sid, n in self.chunks_per_source.items())
        return f"[done] {self.total_chunks} chunks ({per_source}) in index '{self.index_name}'"


def chunk_source(source: DocSource, cfg: RagConfig) -> list[Document]:
    """Fetch one source's documents and chunk them."""
    chunks = chunk_documents(source.fetch(), cfg.chunking)
    print(f"[chunk] {source.spec.id}: {len(chunks)} chunks")
    return chunks


def run_ingest(
    cfg: RagConfig,
    sources: Iterable[DocSource],
    upsert=upsert_chunks,
    store=None,
) -> IngestReport:
    """Run the full ingest path for every source.

    `upsert` and `store` are injectable so a test can run the entire pipeline
    against fakes. In production both defaults apply: chunks go to the real
    Pinecone index named in the config.
    """
    chunks_per_source: dict[str, int] = {}
    all_chunks: list[Document] = []

    for source in sources:
        chunks = chunk_source(source, cfg)
        chunks_per_source[source.spec.id] = len(chunks)
        all_chunks.extend(chunks)

    log_event(
        "ingest_chunked",
        project=cfg.project,
        sources=len(chunks_per_source),
        chunks=len(all_chunks),
    )

    # store=None lets upsert_chunks build the real (index-creating) store.
    upsert(cfg, all_chunks, store=store)

    report = IngestReport(
        chunks_per_source=chunks_per_source,
        total_chunks=len(all_chunks),
        index_name=cfg.pinecone_index_name,
    )
    log_event("ingest_done", project=cfg.project, chunks=report.total_chunks)
    return report

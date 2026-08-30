"""Retriever: question -> top-k chunks with scores + metadata.

This module owns the boundary between LangChain's vector-store types and
everything downstream: answering, confidence, and (in a project) ticket
drafting all work with `RetrievedChunk`, never with raw `Document`s.

The store is passed in as a parameter so unit tests can use a fake with a
canned `similarity_search_with_score`. Real callers get it from
`make_retriever(cfg)`, which pins the query-time embedding model to the same
config value used at ingestion (the two must match or retrieval silently
degrades) and fails loudly if the index doesn't exist.
"""

from dataclasses import dataclass

from langsmith import traceable

from rag_core.config import RagConfig


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    # Cosine similarity from the vector store; higher = closer. NOT a
    # probability — typical "good match" values for text-embedding-3-small
    # sit around 0.4-0.6.
    score: float
    # Which configured source this chunk came from (was "service" when this
    # engine only knew about AWS).
    source_id: str
    source_file: str
    heading: str
    url: str


def retrieve(question: str, store, k: int = 4) -> list[RetrievedChunk]:
    """Embed the question and return the k nearest chunks, best first.

    `store` is anything with `similarity_search_with_score(query, k=k)`
    returning (Document, score) pairs — a PineconeVectorStore in production,
    a fake in tests.
    """
    results = store.similarity_search_with_score(question, k=k)
    chunks = []
    for doc, score in results:
        meta = doc.metadata
        chunks.append(
            RetrievedChunk(
                text=doc.page_content,
                score=score,
                # Metadata was attached at ingestion; default to "" rather
                # than crash if a field is ever missing from a vector.
                source_id=meta.get("source_id", ""),
                source_file=meta.get("source_file", ""),
                heading=meta.get("heading", ""),
                url=meta.get("url", ""),
            )
        )
    return chunks


def make_retriever(cfg: RagConfig, store=None):
    """Bind a retriever to a real vector store for production callers.

    Returns a `retrieve(question, k=...)`-shaped callable whose default k is
    the configured `retrieval.top_k`. The vectorstore module is imported
    lazily so unit tests of this module never touch Pinecone/OpenAI clients.
    """
    if store is None:
        from rag_core.vectorstore.pinecone_store import get_vector_store_for_query

        store = get_vector_store_for_query(cfg)

    default_k = cfg.retrieval.top_k

    # Traced HERE (not on `retrieve`) so the span's inputs are the clean
    # (question, k) pair; `retrieve`'s `store` argument would get serialized
    # into the trace otherwise. No-op unless LANGSMITH_TRACING is on.
    @traceable(name="retrieve")
    def _retrieve(question: str, k: int | None = None) -> list[RetrievedChunk]:
        return retrieve(question, store, k=default_k if k is None else k)

    return _retrieve

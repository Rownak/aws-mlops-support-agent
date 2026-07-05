"""Task 2.1 — Retriever: question -> top-k chunks with scores + metadata.

This module owns the boundary between LangChain's vector-store types and the
rest of the agent: everything downstream (answering, confidence, the Jira
"docs checked" field) works with `RetrievedChunk`, never with raw `Document`s.

The store is passed in as a parameter so unit tests can use a fake with a
canned `similarity_search_with_score`. Real callers get it from
`src.ingest.index.get_vector_store(cfg)`, which pins the query-time embedding
model to the same config value used at ingestion — the two must match or
retrieval silently degrades.
"""

from dataclasses import dataclass

from src.config import Config


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    # Cosine similarity from Pinecone; higher = closer. NOT a probability —
    # typical "good match" values for text-embedding-3-small sit around 0.4–0.6.
    score: float
    service: str
    source_file: str
    heading: str
    url: str


def retrieve(question: str, store, k: int = 4) -> list[RetrievedChunk]:
    """Embed the question and return the k nearest chunks, best first.

    `store` is anything with `similarity_search_with_score(query, k=k)`
    returning (Document, score) pairs — the PineconeVectorStore in production,
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
                # Metadata was attached at ingestion (task 1.3); default to ""
                # rather than crash if a field is ever missing from a vector.
                service=meta.get("service", ""),
                source_file=meta.get("source_file", ""),
                heading=meta.get("heading", ""),
                url=meta.get("url", ""),
            )
        )
    return chunks


def make_retriever(cfg: Config):
    """Convenience for production callers: bind a real Pinecone store.

    Returns a `retrieve(question, k=...)`-shaped callable. Imported lazily so
    unit tests of this module never touch Pinecone/OpenAI clients.
    """
    from src.ingest.index import get_vector_store

    store = get_vector_store(cfg)

    def _retrieve(question: str, k: int = 4) -> list[RetrievedChunk]:
        return retrieve(question, store, k=k)

    return _retrieve

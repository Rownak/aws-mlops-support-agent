"""Pinecone wrapped behind the `Retriever` protocol (design.md phase 7).

Read-only: no ingestion, no index lifecycle — those stay on `PineconeStore`.
Wraps a `PineconeVectorStore` (from `PineconeStore.get_store()`), the same
object `retrieve.py` already queries in production, so this reuses the
identical `similarity_search_with_relevance_scores` call rather than a second
Pinecone code path.
"""

from typing import Any, Dict, Optional

from rag_core.retriever.base import SearchResult


class PineconeRetriever:
    """Cosine similarity search over an existing Pinecone-backed vectorstore.

    `filters` is a Pinecone-native metadata filter (MongoDB-style operators),
    passed straight through to `similarity_search_with_relevance_scores` —
    unused by any caller today, kept so a future filtered-retrieval config
    doesn't need a signature change (claude/docs/phase7_pinecone_scope_decision.md).
    """

    def __init__(
        self,
        vectorstore: Any,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ):
        self._vectorstore = vectorstore
        self.top_k = top_k
        self.filters = filters

    def search(self, query: str, k: int | None = None) -> list[SearchResult]:
        k = k if k is not None else self.top_k

        search_kwargs: Dict[str, Any] = {"k": k}
        if self.filters:
            search_kwargs["filter"] = self.filters

        scored = self._vectorstore.similarity_search_with_relevance_scores(
            query, **search_kwargs
        )

        return [
            SearchResult(
                doc_id=document.metadata["chunk_id"],
                document=document,
                score=float(score),
                score_type="cosine",
            )
            for document, score in scored
        ]

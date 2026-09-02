"""The retriever contract shared by every retrieval technique (design_summary.md).

Concrete retrievers (BM25, dense, HyDE, multi-query, RRF, reranking, Pinecone)
all implement `Retriever` — a protocol, not a base class, so composition
(a reranker wrapping an RRF fusion of BM25 + dense) is just nesting instances,
no inheritance hierarchy to design around.
"""

from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    document: Document
    score: float
    score_type: str  # "bm25" | "cosine" | "rrf" | "rerank_logit"


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[SearchResult]: ...

"""Retriever module for hybrid search, reranking, and confidence scoring."""

from .base import Retriever, SearchResult
from .confidence import DEFAULT_MIN_TOP_SCORE, RetrievalConfidence, assess_confidence
from .hybrid import get_retriever, hybrid_search, mmr_search
from .rerank import (
    BaseReranker,
    CohereReranker,
    CrossEncoderReranker,
    get_reranker,
    resolve_fetch_k,
)
from .retrieve import retrieve, retrieve_scored

__all__ = [
    "Retriever",
    "SearchResult",
    "get_retriever",
    "hybrid_search",
    "mmr_search",
    "BaseReranker",
    "CohereReranker",
    "CrossEncoderReranker",
    "get_reranker",
    "resolve_fetch_k",
    "DEFAULT_MIN_TOP_SCORE",
    "RetrievalConfidence",
    "assess_confidence",
    "retrieve",
    "retrieve_scored",
]

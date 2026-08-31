"""Retriever module for hybrid search, reranking, and confidence scoring."""

from .confidence import DEFAULT_MIN_TOP_SCORE, RetrievalConfidence, assess_confidence
from .hybrid import get_retriever, hybrid_search, mmr_search
from .rerank import (
    BaseReranker,
    CohereReranker,
    CrossEncoderReranker,
    get_reranker,
    resolve_fetch_k,
)
from .retrieve import retrieve

__all__ = [
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
]

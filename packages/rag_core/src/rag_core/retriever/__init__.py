"""Retriever module for hybrid search, reranking, and confidence scoring."""

from .base import Retriever, SearchResult
from .bm25 import BM25Retriever
from .confidence import DEFAULT_MIN_TOP_SCORE, RetrievalConfidence, assess_confidence
from .dense import DenseRetriever
from .fusion import RRFRetriever
from .hybrid import get_retriever, hybrid_search, mmr_search
from .rerank import (
    BiEncoderScorer,
    CohereScorer,
    CrossEncoderScorer,
    RelevanceScorer,
    RerankingRetriever,
    get_reranker,
    resolve_fetch_k,
)
from .retrieve import retrieve, retrieve_scored

__all__ = [
    "Retriever",
    "SearchResult",
    "BM25Retriever",
    "DenseRetriever",
    "RRFRetriever",
    "get_retriever",
    "hybrid_search",
    "mmr_search",
    "RelevanceScorer",
    "CohereScorer",
    "CrossEncoderScorer",
    "BiEncoderScorer",
    "RerankingRetriever",
    "get_reranker",
    "resolve_fetch_k",
    "DEFAULT_MIN_TOP_SCORE",
    "RetrievalConfidence",
    "assess_confidence",
    "retrieve",
    "retrieve_scored",
]

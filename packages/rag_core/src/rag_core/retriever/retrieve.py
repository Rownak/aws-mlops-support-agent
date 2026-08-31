"""
The retrieve step, as a standalone function.

Ported from ``references/ragwire/pipeline/pipeline.py``'s ``RAGWire.retrieve``,
trimmed to what rag_core actually has: metadata-filter extraction
(``auto_filter``, ``_extract_filters_from_query``, ``_build_qdrant_filter``)
depends on ragwire's metadata-extraction/schema system, which rag_core does
not implement, so it is not ported.

Kept as a function rather than a facade method so an agent can call retrieval
as its own step — e.g. to run ``assess_confidence`` on the result and escalate
instead of generating an answer — without going through ``RagCore`` at all.
"""

import logging
from typing import Any, List, Tuple

from rag_core.config import RetrieverConfig
from rag_core.retriever.rerank import get_reranker, resolve_fetch_k

logger = logging.getLogger(__name__)


def retrieve(
    question: str,
    vectorstore: Any,
    cfg: RetrieverConfig,
    top_k: int | None = None,
) -> List[Tuple[Any, float]]:
    """
    Retrieve (document, score) pairs for a question, best first.

    When a reranker is configured (``cfg.rerank``), this fetches a wider
    candidate pool with :func:`resolve_fetch_k`, reranks every candidate
    against the question, and returns the best ``top_k`` — each pair's score
    is then the reranker's relevance score rather than a similarity score.

    Args:
        question: The question to retrieve chunks for
        vectorstore: A LangChain vector store exposing
            ``similarity_search_with_score`` (a Pinecone-backed store in
            production, a fake in tests)
        cfg: The retriever config block (search_type, top_k, rerank)
        top_k: Override ``cfg.top_k`` for this call

    Returns:
        (document, score) pairs, best first — the shape
        :func:`rag_core.retriever.confidence.assess_confidence` and the evals
        runner expect.

    Raises:
        ValueError: If ``cfg.search_type`` is "mmr". MMR selects for
            diversity rather than ranking by similarity, and LangChain's MMR
            search has no scored variant, so it cannot produce the
            (document, score) pairs confidence scoring needs. Use
            :func:`rag_core.retriever.hybrid.mmr_search` directly when
            diversity matters more than a confidence signal.
    """
    if cfg.search_type == "mmr":
        raise ValueError(
            "retrieve() requires a similarity score for confidence scoring, "
            "but MMR search has no scored variant. Use "
            "rag_core.retriever.hybrid.mmr_search directly instead."
        )

    resolved_top_k = top_k or cfg.top_k
    reranker = get_reranker(cfg.rerank)
    candidate_k = resolve_fetch_k(cfg.rerank, resolved_top_k) if reranker else resolved_top_k

    scored = vectorstore.similarity_search_with_score(question, k=candidate_k)

    if reranker and scored:
        documents = [doc for doc, _ in scored]
        before = len(documents)
        reranked = reranker.rerank(question, documents, top_n=resolved_top_k)
        # rerank() stamps its own relevance score into each document's
        # metadata rather than returning it alongside the document, so pull
        # it back out to keep the (document, score) shape callers expect.
        scored = [(doc, doc.metadata.get("rerank_score", 0.0)) for doc in reranked]
        logger.info(f"Reranked {before} candidates to {len(scored)} using {reranker.name}")
    else:
        scored = scored[:resolved_top_k]

    logger.info(f"Retrieved {len(scored)} documents for query: {question[:50]}...")
    return scored

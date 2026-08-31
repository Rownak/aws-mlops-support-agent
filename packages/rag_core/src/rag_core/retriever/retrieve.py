"""
The retrieve step, as standalone functions.

Two entry points, deliberately separate rather than one function with a
shape-changing flag:

- :func:`retrieve` returns plain documents. It is the generic primitive — it
  supports every ``search_type`` including MMR, and no caller is forced to
  reason about what a score means.
- :func:`retrieve_scored` returns (document, score) pairs for the callers that
  genuinely need the number: :func:`rag_core.retriever.confidence.assess_confidence`
  and the evals runner.

Scores come from ``similarity_search_with_relevance_scores``, NOT
``similarity_search_with_score``. The raw method returns whatever the backend's
metric produces — cosine in [-1, 1], dot-product unbounded, Euclidean *distance*
where lower is better — so a threshold like ``min_top_score`` silently means a
different thing per backend, and inverts entirely on a distance metric. The
relevance variant normalizes every backend to [0, 1], higher-is-better, which is
what makes a configured threshold portable.

Ported from ``references/ragwire/pipeline/pipeline.py``'s ``RAGWire.retrieve``,
trimmed to what rag_core actually has: metadata-filter *extraction*
(``auto_filter``, ``_extract_filters_from_query``) depends on ragwire's
metadata-extraction/schema system, which rag_core does not implement, so it is
not ported. Filters supplied by the caller ARE passed through.

Kept as functions rather than facade methods so an agent can call retrieval as
its own step — e.g. to run ``assess_confidence`` on the result and escalate
instead of generating an answer — without going through ``RagCore`` at all.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from langsmith import traceable

from rag_core.config import RetrieverConfig
from rag_core.retriever.hybrid import mmr_search
from rag_core.retriever.rerank import get_reranker, resolve_fetch_k

logger = logging.getLogger(__name__)


@traceable(run_type="retriever", name="rag_core.retrieve_scored")
def retrieve_scored(
    question: str,
    vectorstore: Any,
    cfg: RetrieverConfig,
    top_k: int | None = None,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Tuple[Any, float]]:
    """
    Retrieve (document, relevance_score) pairs for a question, best first.

    Scores are normalized to [0, 1] (0 = dissimilar, 1 = most similar), so a
    configured ``min_top_score`` means the same thing across cosine,
    dot-product and Euclidean backends.

    .. warning::
       When a reranker is configured (``cfg.rerank``), the returned score is
       the **reranker's** relevance score instead — a different quantity on a
       different scale (a cross-encoder logit, often unbounded and sometimes
       negative), not a normalized similarity. ``min_top_score`` is currently
       compared against whichever of the two you get, so a threshold tuned
       without reranking will not mean the same thing with it enabled.

    Args:
        question: The question to retrieve chunks for
        vectorstore: A LangChain vector store exposing
            ``similarity_search_with_relevance_scores`` (a Pinecone-backed
            store in production, a fake in tests)
        cfg: The retriever config block (search_type, top_k, rerank)
        top_k: Override ``cfg.top_k`` for this call
        filters: Backend-native metadata filter, passed through untouched.
            Syntax is your vector store's own (Pinecone uses MongoDB-style
            operators, Qdrant uses its Filter model) — rag_core deliberately
            does not invent a neutral filter DSL for one backend.

    Returns:
        (document, score) pairs, best first — the shape
        :func:`rag_core.retriever.confidence.assess_confidence` and the evals
        runner expect.

    Raises:
        ValueError: If ``cfg.search_type`` is "mmr". MMR selects for diversity
            rather than ranking by similarity and has no scored variant, so it
            cannot produce the pairs confidence scoring needs. Use
            :func:`retrieve` for MMR.
    """
    if cfg.search_type == "mmr":
        raise ValueError(
            "retrieve_scored() requires a relevance score for confidence "
            "scoring, but MMR search has no scored variant. Use retrieve() "
            "instead, which supports MMR."
        )

    resolved_top_k = top_k or cfg.top_k
    reranker = get_reranker(cfg.rerank)
    candidate_k = resolve_fetch_k(cfg.rerank, resolved_top_k) if reranker else resolved_top_k

    search_kwargs: Dict[str, Any] = {"k": candidate_k}
    if filters:
        search_kwargs["filter"] = filters

    scored = vectorstore.similarity_search_with_relevance_scores(question, **search_kwargs)

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


@traceable(run_type="retriever", name="rag_core.retrieve")
def retrieve(
    question: str,
    vectorstore: Any,
    cfg: RetrieverConfig,
    top_k: int | None = None,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """
    Retrieve documents for a question, best first.

    The generic primitive: it supports every ``search_type``, MMR included,
    and hands back plain documents. Each document carries its score in
    ``metadata["score"]`` when the search type produced one (absent for MMR,
    which has no score) — that is a convenience for logging, eval tables and
    debugging, not the typed contract. Call :func:`retrieve_scored` when you
    need the score as a value.

    Args:
        question: The question to retrieve chunks for
        vectorstore: A LangChain vector store
        cfg: The retriever config block (search_type, top_k, rerank)
        top_k: Override ``cfg.top_k`` for this call
        filters: Backend-native metadata filter, passed through untouched
            (see :func:`retrieve_scored` for the portability caveat)

    Returns:
        Documents, best first.
    """
    if cfg.search_type == "mmr":
        # Delegated rather than reimplemented: mmr_search already goes through
        # as_retriever().invoke(), so it carries its own VectorStoreRetriever
        # span and MMR's fetch_k/lambda_mult knobs stay in one place.
        documents = mmr_search(vectorstore, question, k=top_k or cfg.top_k, filters=filters)
        logger.info(f"Retrieved {len(documents)} documents (mmr) for query: {question[:50]}...")
        return documents

    scored = retrieve_scored(question, vectorstore, cfg, top_k=top_k, filters=filters)

    documents = []
    for doc, score in scored:
        # Stamped so the score survives for logging/evals without making it
        # part of this function's return type.
        doc.metadata["score"] = score
        documents.append(doc)
    return documents

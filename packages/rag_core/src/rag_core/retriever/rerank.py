"""
Rerankers for second-stage retrieval.

First-stage retrieval (dense, sparse or hybrid) scores the query and the
document separately, then compares the two vectors. A reranker instead reads
the query and each candidate together and scores the pair directly, which is
far more accurate but far too slow to run over a whole collection. The usual
arrangement, and the one RagCore uses, is to retrieve a wide candidate pool
cheaply and rerank it down to the handful of chunks you actually keep.

Three providers ship with the package:

- ``cross_encoder`` runs a local sentence-transformers model. No API key, no
  network calls after the first download, and it is the default.
- ``cohere`` calls the hosted Cohere Rerank endpoint. Needs ``COHERE_API_KEY``.
- ``bi_encoder`` re-embeds candidates with an ``Embeddings`` model and scores
  cosine(query, document) — cheaper than a cross-encoder, and a useful
  sanity check when pointed at the same model first-stage retrieval used.

``cross_encoder`` and ``cohere`` are optional dependencies: install the one
you want with ``pip install rag-core[rerank]`` or ``pip install rag-core[cohere]``.
``bi_encoder`` needs only an ``Embeddings`` instance the caller already has.

This module also holds ``RerankingRetriever``, which wraps one of the scorers
above behind the ``Retriever`` protocol (``search(query, k) ->
list[SearchResult]``) for use in a `rag_core.retriever.factory.build_retriever`
pipeline tree — see its docstring for how the two shapes relate.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.embeddings import Embeddings

from rag_core.retriever.base import Retriever, SearchResult

logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_COHERE_MODEL = "rerank-v3.5"


class RelevanceScorer:
    """
    Common behaviour for relevance scorers: `Document` in, `Document` out.

    This is the production-facing shape `retrieve.py` calls directly against
    a live vectorstore's results, distinct from the `Retriever` protocol
    (`search(query, k) -> list[SearchResult]`) that `RerankingRetriever`
    below implements by wrapping one of these. Subclasses implement
    :meth:`_score`, which returns one relevance score per document in the
    order it was given. Ordering, truncation and score attachment are
    handled here so every provider behaves identically.
    """

    name = "base"

    def rerank(
        self, query: str, documents: List[Any], top_n: Optional[int] = None
    ) -> List[Any]:
        """
        Reorder documents by relevance to the query.

        Args:
            query: The search query the documents were retrieved for
            documents: Candidate documents from first-stage retrieval
            top_n: How many documents to keep. Keeps all of them if not given.

        Returns:
            Documents sorted by descending relevance, truncated to top_n. Each
            returned document carries its score in ``metadata["rerank_score"]``.
        """
        if not documents:
            return []

        # A single candidate is already in its final order, and scoring it
        # would cost a model call that cannot change the outcome.
        if len(documents) == 1:
            return documents if top_n is None or top_n >= 1 else []

        scores = self._score(query, [d.page_content for d in documents])

        if len(scores) != len(documents):
            raise ValueError(
                f"{self.name} reranker returned {len(scores)} scores for "
                f"{len(documents)} documents"
            )

        ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
        if top_n is not None:
            ranked = ranked[:top_n]

        for doc, score in ranked:
            # Documents come back from the vector store freshly constructed on
            # every query, so annotating metadata here does not leak into the
            # stored payload.
            doc.metadata["rerank_score"] = float(score)

        return [doc for doc, _ in ranked]

    def _score(self, query: str, texts: List[str]) -> List[float]:
        raise NotImplementedError


class CrossEncoderScorer(RelevanceScorer):
    """
    Local cross-encoder reranker backed by sentence-transformers.

    The model is downloaded on first use rather than at construction, so
    building a RagCore instance for ingestion never pays for a model it will
    not use. The package itself is checked eagerly, so a missing dependency
    surfaces at startup instead of on the first query.
    """

    name = "cross_encoder"

    def __init__(
        self,
        model: str = DEFAULT_CROSS_ENCODER_MODEL,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        try:
            import sentence_transformers  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The cross_encoder reranker requires sentence-transformers. "
                "Install it with: pip install rag-core[rerank]"
            ) from exc

        self.model_name = model
        self.batch_size = batch_size
        self.device = device
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            device = self.device
            if device and device != "cpu":
                import torch

                if not torch.cuda.is_available():
                    logger.warning(
                        f"cross_encoder reranker configured for device={device!r} "
                        "but CUDA is not available; falling back to cpu"
                    )
                    device = "cpu"

            logger.info(f"Loading cross-encoder reranker: {self.model_name} (device={device})")
            kwargs: Dict[str, Any] = {}
            if device:
                kwargs["device"] = device
            self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model

    def _score(self, query: str, texts: List[str]) -> List[float]:
        model = self._load()
        pairs = [(query, text) for text in texts]
        scores = model.predict(pairs, batch_size=self.batch_size)
        return [float(s) for s in scores]


class BiEncoderScorer(RelevanceScorer):
    """
    Re-scores candidates with an embedding model: cosine(query, document).

    A cross-encoder reads the query and document together; a bi-encoder
    embeds them separately, same as first-stage dense retrieval — so pointed
    at the same embedding model the first-stage retriever already used, this
    should show near-zero gain. That is the intended sanity check, not a bug
    (design.md 3.3): a genuine improvement means the reranker is using a
    stronger or differently-tuned model, not just "a second look".
    """

    name = "bi_encoder"

    def __init__(self, embeddings: Embeddings):
        self._embeddings = embeddings

    def _score(self, query: str, texts: List[str]) -> List[float]:
        query_vector = np.array(self._embeddings.embed_query(query))
        query_vector = query_vector / max(np.linalg.norm(query_vector), 1e-12)

        doc_vectors = np.array(self._embeddings.embed_documents(texts))
        norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        doc_vectors = doc_vectors / norms

        return [float(s) for s in doc_vectors @ query_vector]


class CohereScorer(RelevanceScorer):
    """
    Hosted reranker backed by the Cohere Rerank endpoint.

    Cohere returns only the documents it ranked, so scores are mapped back onto
    the original positions before sorting. Any document Cohere omits keeps a
    score low enough to sort last rather than being silently dropped, which
    keeps the contract of returning every input document intact.
    """

    name = "cohere"

    def __init__(
        self,
        model: str = DEFAULT_COHERE_MODEL,
        api_key: Optional[str] = None,
    ):
        try:
            import cohere
        except ImportError as exc:
            raise ImportError(
                "The cohere reranker requires the cohere SDK. "
                "Install it with: pip install rag-core[cohere]"
            ) from exc

        import os

        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise ValueError(
                "The cohere reranker needs an API key. Set COHERE_API_KEY in "
                "your environment or .env file, or pass api_key in the config."
            )

        self.model_name = model
        self._client = cohere.ClientV2(api_key=key)

    def _score(self, query: str, texts: List[str]) -> List[float]:
        response = self._client.rerank(
            model=self.model_name,
            query=query,
            documents=texts,
            top_n=len(texts),
        )

        scores = [float("-inf")] * len(texts)
        for result in response.results:
            scores[result.index] = float(result.relevance_score)
        return scores


class RerankingRetriever:
    """
    A `Retriever` that reranks an inner retriever's candidates.

    Wraps an inner `Retriever` and a `RelevanceScorer`, translating
    `SearchResult <-> Document` at this boundary — the scorers above stay
    `Document`-in/out (what `retrieve.py` needs against a live vectorstore)
    and know nothing about `SearchResult`; this class is the only place the
    two shapes meet. Reuses the scorer's `_score()` via `rerank()` rather
    than duplicating provider logic.

    `candidate_k` must exceed `top_k` or reranking has nothing to reorder —
    it asks the inner retriever for a wide pool, then narrows to `top_k`.
    """

    def __init__(
        self,
        inner: Retriever,
        scorer: RelevanceScorer,
        candidate_k: int,
        top_k: int = 10,
    ):
        if candidate_k < top_k:
            raise ValueError(
                f"candidate_k ({candidate_k}) must be >= top_k ({top_k}), "
                "or reranking has nothing to reorder"
            )
        self._inner = inner
        self._scorer = scorer
        self.candidate_k = candidate_k
        self.top_k = top_k

    def search(self, query: str, k: int | None = None) -> List[SearchResult]:
        k = k if k is not None else self.top_k
        # Ask the inner retriever for the wider candidate pool, not k — a
        # parent requests the depth it needs (design.md's depth rule), and
        # candidate_k is that depth here regardless of what's ultimately kept.
        candidates = self._inner.search(query, k=self.candidate_k)

        # Every retriever's SearchResult.document carries doc_id in metadata
        # (bm25.py, dense.py) — used here rather than page_content identity,
        # which two distinct documents could share.
        documents = [c.document for c in candidates]
        reranked = self._scorer.rerank(query, documents, top_n=k)

        return [
            SearchResult(
                doc_id=doc.metadata["doc_id"],
                document=doc,
                score=doc.metadata["rerank_score"],
                score_type="rerank_logit",
            )
            for doc in reranked
        ]


PROVIDERS = {
    "cross_encoder": CrossEncoderScorer,
    "cohere": CohereScorer,
    # bi_encoder is deliberately absent: it needs a pre-built Embeddings
    # instance (a resource, not a primitive config value), so it cannot be
    # constructed via **kwargs-from-dict below. rag_bench_eval's
    # resources.get_reranker() resolves the embeddings resource and
    # constructs BiEncoderScorer directly instead of going through here.
}


def get_reranker(config: Optional[Dict[str, Any]]) -> Optional[RelevanceScorer]:
    """
    Build a reranker from a ``retriever.rerank`` config block.

    Returns None when reranking is not configured or is explicitly disabled,
    which is what keeps this feature free for everyone who does not use it.

    Args:
        config: The ``retriever.rerank`` mapping, or None

    Returns:
        A reranker instance, or None when reranking is off

    Raises:
        ValueError: If the provider is unknown

    Example:
        >>> get_reranker({"provider": "cross_encoder"})  # doctest: +SKIP
        <CrossEncoderScorer ...>
        >>> get_reranker(None) is None
        True
    """
    if not config:
        return None

    # Presence of the block is enough to turn reranking on. "enabled: false"
    # exists so a config can keep its tuned settings while switching it off.
    if not config.get("enabled", True):
        return None

    provider = config.get("provider", "cross_encoder")
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown rerank provider: '{provider}'. "
            f"Available: {', '.join(sorted(PROVIDERS))}"
        )

    kwargs = {k: v for k, v in config.items() if k not in ("enabled", "provider", "fetch_k")}
    return PROVIDERS[provider](**kwargs)


def resolve_fetch_k(config: Optional[Dict[str, Any]], top_k: int) -> int:
    """
    Decide how many candidates first-stage retrieval should return.

    Reranking can only reorder what it is given, so the candidate pool has to
    be wider than the final result set for it to have anything to do. The
    default of four times top_k is wide enough to matter and small enough that
    a local cross-encoder stays fast.

    Args:
        config: The ``retriever.rerank`` mapping, or None
        top_k: The number of documents the caller ultimately wants

    Returns:
        The candidate count to request from the vector store

    Example:
        >>> resolve_fetch_k({"provider": "cohere"}, top_k=5)
        20
        >>> resolve_fetch_k({"fetch_k": 50}, top_k=5)
        50
    """
    if not config:
        return top_k

    fetch_k = config.get("fetch_k")
    if fetch_k is None:
        fetch_k = max(4 * top_k, 20)

    # Fetching fewer candidates than the caller asked to keep would make the
    # reranker silently shrink the result set.
    return max(int(fetch_k), top_k)

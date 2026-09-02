"""Dense retrieval: embed the corpus once, cosine over a numpy matrix
(design_summary.md).

No chunking — NFCorpus qrels are document-level, so `Doc.content` embeds
whole (design.md §8 Q5). Corpus-agnostic: this module takes plain
(doc_id, text) pairs, same contract as BM25Retriever.
"""

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_core.retriever.base import SearchResult


_EMBED_BATCH_SIZE = 100


class DenseRetriever:
    """Cosine similarity over a fixed corpus, embedded once at construction."""

    def __init__(
        self,
        corpus: dict[str, str],
        embeddings: Embeddings,
        top_k: int = 10,
    ):
        self.top_k = top_k
        self._embeddings = embeddings
        self._doc_ids = list(corpus.keys())

        texts = list(corpus.values())
        embedded: list[list[float]] = []
        # A single request over the whole corpus can overwhelm a local Ollama
        # server (observed: connection reset partway through 3,633 docs) —
        # batching keeps each request small regardless of provider.
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            embedded.extend(embeddings.embed_documents(texts[i : i + _EMBED_BATCH_SIZE]))
        vectors = np.array(embedded)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._normalized = vectors / norms

    def search(self, query: str, k: int | None = None) -> list[SearchResult]:
        k = k if k is not None else self.top_k
        query_vector = np.array(self._embeddings.embed_query(query))
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            query_norm = 1.0
        query_vector = query_vector / query_norm

        scores = self._normalized @ query_vector
        top_indices = np.argsort(scores)[::-1][:k]

        return [
            SearchResult(
                doc_id=self._doc_ids[i],
                document=Document(page_content="", metadata={"doc_id": self._doc_ids[i]}),
                score=float(scores[i]),
                score_type="cosine",
            )
            for i in top_indices
        ]

"""Lexical retrieval over `rank_bm25`, in-memory (design_summary.md).

Deliberately untuned tokenizer — lowercase + whitespace/punctuation split, no
stemming or stopword removal — so this stays a straightforward BM25 baseline
rather than a second thing to tune before the nDCG@10 gate (~0.32) is trusted.
"""

import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag_core.retriever.base import SearchResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """BM25 over a fixed corpus of (doc_id, text) pairs, held in memory."""

    def __init__(
        self,
        corpus: dict[str, str],
        k1: float = 1.5,
        b: float = 0.75,
        top_k: int = 10,
    ):
        self.top_k = top_k
        self._corpus = corpus
        self._doc_ids = list(corpus.keys())
        tokenized = [_tokenize(text) for text in corpus.values()]
        # k1/b tune term-frequency saturation and length normalization;
        # rank_bm25 takes them as constructor args on the Okapi variant.
        self._bm25 = BM25Okapi(tokenized, k1=k1, b=b)

    def search(self, query: str, k: int | None = None) -> list[SearchResult]:
        k = k if k is not None else self.top_k
        scores = self._bm25.get_scores(_tokenize(query))
        pairs = zip(self._doc_ids, scores, strict=True)
        ranked = sorted(pairs, key=lambda pair: pair[1], reverse=True)[:k]
        return [
            SearchResult(
                doc_id=doc_id,
                document=Document(
                    page_content=self._corpus[doc_id], metadata={"doc_id": doc_id}
                ),
                score=float(score),
                score_type="bm25",
            )
            for doc_id, score in ranked
        ]

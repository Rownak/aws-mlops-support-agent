"""BEIR-shaped dataset contract: Doc, Query, Qrels.

No network imports — the metric and dataset tests construct these directly,
without a download.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    text: str

    @property
    def content(self) -> str:
        # BEIR corpora split title/text; retrievers index the concatenation.
        if self.title:
            return f"{self.title} {self.text}"
        return self.text


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


# query_id -> {doc_id: relevance grade (0/1/2)}
Qrels = dict[str, dict[str, int]]

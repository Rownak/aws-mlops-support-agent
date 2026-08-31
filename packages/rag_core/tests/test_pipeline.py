"""Tests for RagCore.retrieve_with_confidence / query / aquery.

RagCore.__init__ builds real provider clients (embeddings, LLM, Pinecone), so
these tests bypass it with __new__ and inject fakes directly, exercising only
the retrieve -> confidence -> generate composition.
"""

import asyncio

from rag_core.config import RetrieverConfig, VectorStoreConfig
from rag_core.generation.generator import AnswerGenerator
from rag_core.pipeline import RagCore


class _FakeDoc:
    def __init__(self, text, source="doc.md"):
        self.page_content = text
        self.metadata = {"source": source}


class _FakeVectorStore:
    def __init__(self, scored_results):
        self.scored_results = scored_results

    def similarity_search_with_score(self, query, k):
        return self.scored_results[:k]


class _FakeStore:
    def __init__(self, vectorstore):
        self._vectorstore = vectorstore

    def get_store(self, use_sparse=False):
        return self._vectorstore


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def invoke(self, messages):
        return _FakeResponse(self.reply)

    async def ainvoke(self, messages):
        return _FakeResponse(self.reply)


def _rag_core(scored_results, reply="cited [1].", min_top_score=0.35):
    rag = RagCore.__new__(RagCore)
    rag.config = type(
        "Cfg",
        (),
        {
            "vectorstore": VectorStoreConfig(),
            "retriever": RetrieverConfig(search_type="similarity", top_k=5, min_top_score=min_top_score),
        },
    )()
    rag.store = _FakeStore(_FakeVectorStore(scored_results))
    rag.generator = AnswerGenerator(llm=_FakeLLM(reply))
    return rag


def _scored(scores):
    return [(_FakeDoc(f"text {i}", f"doc{i}.md"), score) for i, score in enumerate(scores)]


def test_retrieve_with_confidence_returns_documents_and_verdict():
    rag = _rag_core(_scored([0.6, 0.5]))

    documents, confidence = rag.retrieve_with_confidence("question")

    assert len(documents) == 2
    assert confidence.is_confident
    assert confidence.top_score == 0.6


def test_retrieve_with_confidence_flags_weak_matches():
    rag = _rag_core(_scored([0.1]))

    documents, confidence = rag.retrieve_with_confidence("question")

    assert not confidence.is_confident
    assert "weak" in confidence.reason


def test_query_generates_an_answer_regardless_of_confidence():
    rag = _rag_core(_scored([0.1]))  # low confidence

    answer = rag.query("question")

    assert not answer.refused  # query() always generates; escalation is the caller's job
    assert answer.sources == ["doc0.md"]


def test_query_respects_k_override():
    rag = _rag_core(_scored([0.9, 0.8, 0.7]))

    documents, _ = rag.retrieve_with_confidence("question", k=1)

    assert len(documents) == 1


def test_aquery_generates_an_answer():
    rag = _rag_core(_scored([0.6]))

    answer = asyncio.run(rag.aquery("question"))

    assert not answer.refused

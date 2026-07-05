"""Tests for task 2.1 — retriever mapping, no network involved."""

from langchain_core.documents import Document

from src.agent.retriever import RetrievedChunk, retrieve


class FakeStore:
    """Stands in for PineconeVectorStore; returns canned (Document, score) pairs."""

    def __init__(self, results):
        self.results = results
        self.calls = []

    def similarity_search_with_score(self, query, k=4):
        self.calls.append((query, k))
        return self.results[:k]


def _doc(text, **meta):
    return Document(page_content=text, metadata=meta)


def test_maps_documents_to_chunks_in_order():
    store = FakeStore(
        [
            (
                _doc(
                    "Use the env section.",
                    service="codebuild",
                    source_file="build-env-ref.md",
                    heading="Environment variables",
                    url="https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref.html",
                ),
                0.61,
            ),
            (
                _doc(
                    "Pipelines have stages.",
                    service="codepipeline",
                    source_file="concepts.md",
                    heading="Concepts",
                    url="https://docs.aws.amazon.com/codepipeline/latest/userguide/concepts.html",
                ),
                0.42,
            ),
        ]
    )

    chunks = retrieve("how do I set env vars?", store)

    assert [c.score for c in chunks] == [0.61, 0.42]  # best first, order preserved
    top = chunks[0]
    assert top == RetrievedChunk(
        text="Use the env section.",
        score=0.61,
        service="codebuild",
        source_file="build-env-ref.md",
        heading="Environment variables",
        url="https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref.html",
    )


def test_passes_question_and_k_to_store():
    store = FakeStore([])
    retrieve("some question", store, k=7)
    assert store.calls == [("some question", 7)]


def test_missing_metadata_defaults_to_empty_strings():
    store = FakeStore([(_doc("orphan text"), 0.5)])
    (chunk,) = retrieve("q", store)
    assert chunk.service == ""
    assert chunk.source_file == ""
    assert chunk.heading == ""
    assert chunk.url == ""


def test_no_results_returns_empty_list():
    assert retrieve("q", FakeStore([])) == []

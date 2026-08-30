"""Tests for retriever mapping, no network involved."""

from langchain_core.documents import Document
from rag_core.retrieval.retriever import RetrievedChunk, retrieve


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
                    source_id="codebuild",
                    source_file="build-env-ref.md",
                    heading="Environment variables",
                    url="https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref.html",
                ),
                0.61,
            ),
            (
                _doc(
                    "Pipelines have stages.",
                    source_id="codepipeline",
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
        source_id="codebuild",
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
    assert chunk.source_id == ""
    assert chunk.source_file == ""
    assert chunk.heading == ""
    assert chunk.url == ""


def test_no_results_returns_empty_list():
    assert retrieve("q", FakeStore([])) == []


def test_make_retriever_defaults_k_to_the_configured_top_k():
    """A caller that passes no k gets retrieval.top_k, not a hardcoded 4."""
    from rag_core.config import ChunkingConfig, RagConfig, RetrievalConfig
    from rag_core.retrieval.retriever import make_retriever

    cfg = RagConfig(
        openai_api_key="k",
        openai_chat_model="m",
        openai_embedding_model="text-embedding-3-small",
        pinecone_api_key="k",
        pinecone_index_name="i",
        pinecone_index_metric="cosine",
        aws_region="us-east-1",
        project="p",
        chunking=ChunkingConfig(),
        retrieval=RetrievalConfig(top_k=9),
        sources=(),
    )
    store = FakeStore([])
    retriever = make_retriever(cfg, store=store)

    retriever("q")
    retriever("q", k=2)  # an explicit k still wins
    assert store.calls == [("q", 9), ("q", 2)]

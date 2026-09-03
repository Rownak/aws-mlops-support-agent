from langchain_core.documents import Document

from rag_core.retriever.pinecone import PineconeRetriever


class _StubVectorstore:
    """Records the last search call and returns a fixed, pre-sorted result."""

    def __init__(self, results):
        self._results = results
        self.last_call = None

    def similarity_search_with_relevance_scores(self, query, **kwargs):
        self.last_call = (query, kwargs)
        return self._results


def _doc(chunk_id, text="body"):
    return Document(page_content=text, metadata={"chunk_id": chunk_id})


def test_search_returns_well_formed_search_results():
    stub = _StubVectorstore([(_doc("a_0"), 0.9), (_doc("a_1"), 0.4)])
    retriever = PineconeRetriever(vectorstore=stub, top_k=5)

    results = retriever.search("how do I cache?")

    assert [r.doc_id for r in results] == ["a_0", "a_1"]
    assert [r.score for r in results] == [0.9, 0.4]
    assert all(r.score_type == "cosine" for r in results)
    assert results[0].document.page_content == "body"


def test_search_uses_top_k_default_and_k_override():
    stub = _StubVectorstore([])
    retriever = PineconeRetriever(vectorstore=stub, top_k=7)

    retriever.search("q")
    assert stub.last_call[1]["k"] == 7

    retriever.search("q", k=3)
    assert stub.last_call[1]["k"] == 3


def test_search_passes_filters_through_when_set():
    stub = _StubVectorstore([])
    retriever = PineconeRetriever(vectorstore=stub, filters={"source": {"$eq": "x.md"}})

    retriever.search("q")

    assert stub.last_call[1]["filter"] == {"source": {"$eq": "x.md"}}


def test_search_omits_filter_kwarg_when_none():
    stub = _StubVectorstore([])
    retriever = PineconeRetriever(vectorstore=stub)

    retriever.search("q")

    assert "filter" not in stub.last_call[1]

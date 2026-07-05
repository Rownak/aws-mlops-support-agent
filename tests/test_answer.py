"""Tests for task 2.2 — pure formatting only; the LLM call is checked
manually via `python -m src.agent.ask` (small evals over mocked-API tests)."""

from src.agent.answer import format_context, format_sources
from src.agent.retriever import RetrievedChunk


def _chunk(n):
    return RetrievedChunk(
        text=f"excerpt text {n}",
        score=0.5,
        service="codebuild",
        source_file=f"file{n}.md",
        heading=f"Heading {n}",
        url=f"https://docs.aws.amazon.com/{n}.html",
    )


def test_context_numbers_and_labels_each_excerpt():
    context = format_context([_chunk(1), _chunk(2)])
    assert "[1] service: codebuild | section: Heading 1" in context
    assert "[2] service: codebuild | section: Heading 2" in context
    assert "url: https://docs.aws.amazon.com/1.html" in context
    assert "excerpt text 2" in context
    # [1] must come before [2] — citation numbers reflect retrieval rank.
    assert context.index("[1]") < context.index("[2]")


def test_sources_numbering_matches_context_numbering():
    chunks = [_chunk(1), _chunk(2)]
    sources = format_sources(chunks)
    assert sources.startswith("Sources:")
    assert "[1] Heading 1 — https://docs.aws.amazon.com/1.html" in sources
    assert "[2] Heading 2 — https://docs.aws.amazon.com/2.html" in sources


def test_empty_chunks_produce_empty_context():
    assert format_context([]) == ""

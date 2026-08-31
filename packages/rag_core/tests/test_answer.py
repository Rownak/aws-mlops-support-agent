"""Tests for the Answer/Citation value objects and generation helpers."""

from rag_core.generation.answer import REFUSAL_SENTINEL, Answer, Citation
from rag_core.generation.generator import (
    build_context,
    citation_coverage,
    parse_citations,
)


class _FakeDoc:
    def __init__(self, text, source="doc.md"):
        self.page_content = text
        self.metadata = {"source": source}


def test_answer_bool_reflects_refused():
    assert not Answer(text="x", refused=True)
    assert Answer(text="x", refused=False)


def test_answer_sources_dedupes_in_first_seen_order():
    citations = [
        Citation(1, "a.md", "text"),
        Citation(2, "b.md", "text"),
        Citation(3, "a.md", "text"),
    ]
    answer = Answer(text="x", citations=citations)
    assert answer.sources == ["a.md", "b.md"]


def test_answer_formatted_appends_sources():
    answer = Answer(text="Some claim [1].", citations=[Citation(1, "a.md", "text")])
    formatted = answer.formatted()
    assert "Some claim [1]." in formatted
    assert "[1] a.md" in formatted


def test_answer_formatted_without_citations_is_just_text():
    answer = Answer(text="plain text")
    assert answer.formatted() == "plain text"


def test_citation_snippet_truncates_long_text():
    citation = Citation(1, "a.md", "x " * 200)
    assert citation.snippet.endswith("...")
    assert len(citation.snippet) <= 203


def test_build_context_stops_at_budget():
    docs = [_FakeDoc("a" * 100, "a.md"), _FakeDoc("b" * 100, "b.md")]
    context, used = build_context(docs, max_context_chars=150)
    assert len(used) == 1
    assert used[0].metadata["source"] == "a.md"


def test_build_context_includes_all_when_budget_is_generous():
    docs = [_FakeDoc("a", "a.md"), _FakeDoc("b", "b.md")]
    context, used = build_context(docs, max_context_chars=10000)
    assert len(used) == 2
    assert "[1]" in context and "[2]" in context


def test_parse_citations_resolves_valid_markers():
    docs = [_FakeDoc("text one", "a.md"), _FakeDoc("text two", "b.md")]
    text, citations = parse_citations("Claim one [1]. Claim two [2].", docs)
    assert [c.source for c in citations] == ["a.md", "b.md"]


def test_parse_citations_strips_dangling_markers():
    docs = [_FakeDoc("text one", "a.md")]
    text, citations = parse_citations("Claim [1] and a bad one [7].", docs)
    assert "[7]" not in text
    assert len(citations) == 1


def test_citation_coverage_fraction():
    assert citation_coverage("Sentence one [1]. Sentence two.") == 0.5
    assert citation_coverage("Cited [1].") == 1.0
    assert citation_coverage("") == 0.0


def test_refusal_sentinel_is_a_plain_string():
    assert REFUSAL_SENTINEL == "INSUFFICIENT_CONTEXT"

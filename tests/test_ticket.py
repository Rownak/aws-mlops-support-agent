"""Tests for task 3.5 — pure ticket-draft builder."""

from src.agent.ticket import build_ticket_draft
from src.rag.confidence import RetrievalConfidence
from src.rag.retriever import RetrievedChunk


def _chunk(heading, url):
    return RetrievedChunk(
        text="t", score=0.3, service="codebuild", source_file="f.md", heading=heading, url=url
    )


def _state(**overrides):
    state = {
        "question": "Why does my build fail?",
        "chunks": [],
        "answer": None,
        "attempts": 1,
        "confidence": None,
        "resolved": False,
        "user_action": None,
        "ticket_draft": None,
    }
    state.update(overrides)
    return state


def test_draft_contains_question_answer_and_confidence():
    conf = RetrievalConfidence(
        top_score=0.2, score_gap=0.01, is_confident=False, reason="best match is weak"
    )
    draft = build_ticket_draft(_state(answer="Try checking IAM.", confidence=conf, attempts=2))

    assert "Why does my build fail?" in draft.summary
    assert "Try checking IAM." in draft.description
    assert "best match is weak" in draft.description
    assert "Retrieval attempts: 2" in draft.description
    assert draft.suggested_next_steps  # never empty


def test_docs_checked_deduped_in_rank_order():
    chunks = [
        _chunk("Env vars", "https://docs.aws/a.html"),
        _chunk("Buildspec", "https://docs.aws/b.html"),
        # Same page retrieved twice (two chunks of one section) -> one line.
        _chunk("Env vars", "https://docs.aws/a.html"),
    ]
    draft = build_ticket_draft(_state(chunks=chunks))
    assert draft.docs_checked == [
        "Env vars — https://docs.aws/a.html",
        "Buildspec — https://docs.aws/b.html",
    ]


def test_no_answer_case_is_explicit():
    draft = build_ticket_draft(_state())
    assert "No automated answer was produced." in draft.description


def test_render_is_printable_and_complete():
    text = build_ticket_draft(_state(chunks=[_chunk("H", "U")])).render()
    for section in ("Summary:", "Description:", "Docs checked:", "Suggested next steps:"):
        assert section in text

"""Tests for task 3.4 — routing functions are pure, no graph needed."""

from langgraph.graph import END

from src.agent.graph import MAX_ATTEMPTS, route_after_confirm, route_after_retrieve
from src.rag.confidence import RetrievalConfidence


def _conf(is_confident):
    return RetrievalConfidence(
        top_score=0.5, score_gap=0.1, is_confident=is_confident, reason="test"
    )


def _state(**overrides):
    state = {
        "question": "q",
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


def test_confident_retrieval_goes_to_answer():
    assert route_after_retrieve(_state(confidence=_conf(True))) == "answer"


def test_low_confidence_goes_to_escalate():
    assert route_after_retrieve(_state(confidence=_conf(False))) == "escalate"


def test_missing_confidence_goes_to_escalate():
    # Defensive: retrieve always sets confidence, but None must not answer.
    assert route_after_retrieve(_state(confidence=None)) == "escalate"


def test_resolved_ends_the_graph():
    assert route_after_confirm(_state(user_action="resolved")) == END


def test_ticket_request_escalates():
    assert route_after_confirm(_state(user_action="ticket")) == "escalate"


def test_retry_loops_back_while_attempts_remain():
    assert MAX_ATTEMPTS >= 2  # guard: test below assumes attempts=1 is under the cap
    assert route_after_confirm(_state(user_action="retry", attempts=1)) == "retrieve"


def test_retry_at_cap_escalates():
    assert route_after_confirm(_state(user_action="retry", attempts=MAX_ATTEMPTS)) == "escalate"

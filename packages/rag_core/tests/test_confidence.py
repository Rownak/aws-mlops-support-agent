"""Tests for the pure-function confidence heuristic."""

import pytest
from rag_core.retriever.confidence import DEFAULT_MIN_TOP_SCORE, assess_confidence


class _FakeDoc:
    def __init__(self, source="f"):
        self.metadata = {"source": source}


def _scored(score):
    return (_FakeDoc(), score)


def test_empty_retrieval_is_not_confident():
    result = assess_confidence([])
    assert not result.is_confident
    assert result.top_score == 0.0
    assert result.score_gap == 0.0
    assert "no chunks" in result.reason


def test_weak_top_score_is_not_confident():
    result = assess_confidence([_scored(0.20), _scored(0.18)])
    assert not result.is_confident
    assert "weak" in result.reason
    assert result.top_score == 0.20


def test_healthy_scores_are_confident():
    result = assess_confidence([_scored(0.55), _scored(0.40)])
    assert result.is_confident
    assert result.top_score == 0.55
    assert result.score_gap == pytest.approx(0.15)


def test_exactly_at_threshold_is_confident():
    # Boundary is inclusive: only strictly-below the threshold fails.
    assert assess_confidence([_scored(DEFAULT_MIN_TOP_SCORE)]).is_confident


def test_single_chunk_has_zero_gap():
    result = assess_confidence([_scored(0.5)])
    assert result.score_gap == 0.0
    assert result.is_confident


# --- the threshold is a caller-supplied parameter, not a constant ---


def test_non_default_threshold_flips_the_verdict():
    """The same chunk is confident under a low bar and not under a high one."""
    scored = [_scored(0.40)]
    assert assess_confidence(scored, min_top_score=0.30).is_confident
    assert not assess_confidence(scored, min_top_score=0.50).is_confident


def test_reason_quotes_the_threshold_actually_used():
    result = assess_confidence([_scored(0.40)], min_top_score=0.50)
    assert "0.5" in result.reason


def test_a_stricter_threshold_still_reports_the_real_scores():
    result = assess_confidence([_scored(0.40), _scored(0.10)], min_top_score=0.90)
    assert not result.is_confident
    assert result.top_score == 0.40
    assert result.score_gap == pytest.approx(0.30)

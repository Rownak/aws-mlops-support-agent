"""Tests for task 2.3 — pure-function confidence heuristic."""

import pytest

from src.rag.confidence import MIN_TOP_SCORE, assess_confidence
from src.rag.retriever import RetrievedChunk


def _chunk(score):
    return RetrievedChunk(text="t", score=score, service="s", source_file="f", heading="h", url="u")


def test_empty_retrieval_is_not_confident():
    result = assess_confidence([])
    assert not result.is_confident
    assert result.top_score == 0.0
    assert result.score_gap == 0.0
    assert "no chunks" in result.reason


def test_weak_top_score_is_not_confident():
    result = assess_confidence([_chunk(0.20), _chunk(0.18)])
    assert not result.is_confident
    assert "weak" in result.reason
    assert result.top_score == 0.20


def test_healthy_scores_are_confident():
    result = assess_confidence([_chunk(0.55), _chunk(0.40)])
    assert result.is_confident
    assert result.top_score == 0.55
    assert result.score_gap == pytest.approx(0.15)


def test_exactly_at_threshold_is_confident():
    # Boundary is inclusive: only strictly-below the threshold fails.
    assert assess_confidence([_chunk(MIN_TOP_SCORE)]).is_confident


def test_single_chunk_has_zero_gap():
    result = assess_confidence([_chunk(0.5)])
    assert result.score_gap == 0.0
    assert result.is_confident

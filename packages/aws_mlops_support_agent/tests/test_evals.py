"""Offline tests for the eval runner's pure parts.

A live eval needs a real index; these tests inject a fake retriever with
canned (document, score) pairs. Nothing here mentions a specific corpus — the
runner takes its router as a parameter, which is what keeps it generic.
"""

from aws_mlops_support_agent.evals.runner import (
    EvalCase,
    evaluate_case,
    format_results_table,
    summarize,
)
from rag_core.retriever.confidence import DEFAULT_MIN_TOP_SCORE

K = 4
CONFIDENT = DEFAULT_MIN_TOP_SCORE + 0.15  # comfortably above the threshold
WEAK = DEFAULT_MIN_TOP_SCORE - 0.10  # comfortably below


class _FakeDoc:
    def __init__(self, source):
        self.metadata = {"source": source}


def _scored(score, source="codebuild/build-caching.md"):
    return (_FakeDoc(source), score)


def _case(expected_files=("codebuild/build-caching.md",), should_escalate=False):
    return EvalCase(
        question="How do I cache builds?",
        expected_files=expected_files,
        should_escalate=should_escalate,
        notes="test case",
    )


def _retriever(scored_documents):
    return lambda question, k=K: scored_documents


def _escalates(confidence) -> bool:
    """Stand-in for a project's real router: escalate when not confident."""
    return not confidence.is_confident


def _evaluate(case, scored_documents):
    return evaluate_case(case, _retriever(scored_documents), K, DEFAULT_MIN_TOP_SCORE, _escalates)


def test_hit_when_expected_file_retrieved():
    result = _evaluate(_case(), [_scored(CONFIDENT)])
    assert result.hit is True
    assert result.escalated is False
    assert result.passed


def test_miss_when_only_other_files_retrieved():
    result = _evaluate(_case(), [_scored(CONFIDENT, source="unrelated.md")])
    assert result.hit is False
    assert not result.passed


def test_off_corpus_case_wants_escalation():
    case = _case(expected_files=(), should_escalate=True)
    weak = _evaluate(case, [_scored(WEAK)])
    assert weak.hit is None  # hit@k not applicable without labels
    assert weak.escalated is True
    assert weak.passed

    # Threshold too permissive -> confident on junk -> case fails.
    strong = _evaluate(case, [_scored(CONFIDENT)])
    assert strong.escalated is False
    assert not strong.passed


def test_on_corpus_case_that_escalates_fails_even_with_hit():
    # Right file retrieved but below threshold: routing is still wrong.
    result = _evaluate(_case(), [_scored(WEAK)])
    assert result.hit is True
    assert result.escalated is True
    assert not result.passed


def test_threshold_is_a_parameter_not_a_constant():
    """The same retrieval routes differently under a different threshold."""
    case = _case()
    lenient = evaluate_case(case, _retriever([_scored(0.40)]), K, 0.30, _escalates)
    strict = evaluate_case(case, _retriever([_scored(0.40)]), K, 0.50, _escalates)
    assert lenient.escalated is False
    assert strict.escalated is True


def test_runner_uses_the_injected_router():
    """A router that always escalates must be obeyed, however good the score."""
    result = evaluate_case(
        _case(), _retriever([_scored(CONFIDENT)]), K, DEFAULT_MIN_TOP_SCORE, lambda c: True
    )
    assert result.escalated is True


def test_summary_math():
    on_hit = _evaluate(_case(), [_scored(CONFIDENT)])
    on_miss = _evaluate(_case(), [_scored(CONFIDENT, source="unrelated.md")])
    off_ok = _evaluate(_case(expected_files=(), should_escalate=True), [_scored(WEAK)])
    summary = summarize([on_hit, on_miss, off_ok])
    assert summary.on_corpus_total == 2
    assert summary.on_corpus_hits == 1
    assert summary.escalation_total == 3
    assert summary.escalation_correct == 3  # both on-corpus answered, off escalated


def test_table_has_a_row_per_case_and_lists_failures():
    ok = _evaluate(_case(), [_scored(CONFIDENT)])
    bad = _evaluate(_case(), [_scored(CONFIDENT, source="unrelated.md")])
    table = format_results_table([ok, bad], summarize([ok, bad]), K, DEFAULT_MIN_TOP_SCORE)
    assert table.count("| How do I cache builds?") == 2
    assert "Hit@4 (on-corpus):** 1/2" in table
    # The miss shows what WAS retrieved, for debugging.
    assert "unrelated.md" in table


def test_table_omits_the_failures_section_when_everything_passes():
    ok = _evaluate(_case(), [_scored(CONFIDENT)])
    table = format_results_table([ok], summarize([ok]), K, DEFAULT_MIN_TOP_SCORE)
    assert "### Failures" not in table

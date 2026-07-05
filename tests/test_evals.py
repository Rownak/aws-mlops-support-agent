"""Task 5.1 — offline tests for the eval runner's pure parts.

The live eval (`python -m src.evals`) needs Pinecone/OpenAI; these tests
inject a fake retriever with canned chunks, same style as test_graph.py.
"""

from src.evals.dataset import EVAL_CASES, EvalCase
from src.evals.run import evaluate_case, format_results_table, summarize
from src.rag.confidence import MIN_TOP_SCORE
from src.rag.retriever import RetrievedChunk


def _chunk(score, service="codebuild", source_file="build-caching.md"):
    return RetrievedChunk(
        text="Some doc text.",
        score=score,
        service=service,
        source_file=source_file,
        heading="A heading",
        url="https://docs.aws/a.html",
    )


def _case(expected_files=("codebuild/build-caching.md",), should_escalate=False):
    return EvalCase(
        question="How do I cache builds?",
        expected_files=expected_files,
        should_escalate=should_escalate,
        notes="test case",
    )


def _retriever(chunks):
    return lambda question, k=4: chunks


CONFIDENT = MIN_TOP_SCORE + 0.15  # comfortably above the threshold
WEAK = MIN_TOP_SCORE - 0.10  # comfortably below


def test_hit_when_expected_file_retrieved():
    result = evaluate_case(_case(), _retriever([_chunk(CONFIDENT)]))
    assert result.hit is True
    assert result.escalated is False
    assert result.passed


def test_miss_when_only_other_files_retrieved():
    chunks = [_chunk(CONFIDENT, source_file="unrelated.md")]
    result = evaluate_case(_case(), _retriever(chunks))
    assert result.hit is False
    assert not result.passed


def test_same_filename_in_other_service_is_not_a_hit():
    # concepts.md exists in both repos — labels are "service/filename".
    case = _case(expected_files=("codepipeline/concepts.md",))
    chunks = [_chunk(CONFIDENT, service="codebuild", source_file="concepts.md")]
    assert evaluate_case(case, _retriever(chunks)).hit is False


def test_off_corpus_case_wants_escalation():
    case = _case(expected_files=(), should_escalate=True)
    weak = evaluate_case(case, _retriever([_chunk(WEAK)]))
    assert weak.hit is None  # hit@k not applicable without labels
    assert weak.escalated is True
    assert weak.passed

    # Threshold too permissive -> confident on junk -> case fails.
    strong = evaluate_case(case, _retriever([_chunk(CONFIDENT)]))
    assert strong.escalated is False
    assert not strong.passed


def test_on_corpus_case_that_escalates_fails_even_with_hit():
    # Right file retrieved but below threshold: routing is still wrong.
    result = evaluate_case(_case(), _retriever([_chunk(WEAK)]))
    assert result.hit is True
    assert result.escalated is True
    assert not result.passed


def test_summary_math():
    on_hit = evaluate_case(_case(), _retriever([_chunk(CONFIDENT)]))
    on_miss = evaluate_case(_case(), _retriever([_chunk(CONFIDENT, source_file="unrelated.md")]))
    off_ok = evaluate_case(
        _case(expected_files=(), should_escalate=True), _retriever([_chunk(WEAK)])
    )
    summary = summarize([on_hit, on_miss, off_ok])
    assert summary.on_corpus_total == 2
    assert summary.on_corpus_hits == 1
    assert summary.escalation_total == 3
    assert summary.escalation_correct == 3  # both on-corpus routed to answer, off to escalate


def test_table_has_a_row_per_case_and_lists_failures():
    ok = evaluate_case(_case(), _retriever([_chunk(CONFIDENT)]))
    bad = evaluate_case(_case(), _retriever([_chunk(CONFIDENT, source_file="unrelated.md")]))
    table = format_results_table([ok, bad], summarize([ok, bad]))
    assert table.count("| How do I cache builds?") == 2
    assert "Hit@4 (on-corpus):** 1/2" in table
    # The miss shows what WAS retrieved, for debugging.
    assert "codebuild/unrelated.md" in table


def test_dataset_labels_point_at_real_corpus_services():
    # Guard against typos in hand-written labels: every expected file is
    # "service/filename" with a known service slug.
    for case in EVAL_CASES:
        assert (case.expected_files == ()) == case.should_escalate
        for f in case.expected_files:
            service, _, filename = f.partition("/")
            assert service in ("codebuild", "codepipeline"), f
            assert filename.endswith(".md"), f

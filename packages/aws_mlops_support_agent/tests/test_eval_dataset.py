"""Tests for THIS corpus's eval dataset — the corpus-specific half of evals.

The runner's own logic is tested in rag_core; what matters here is that the
hand-written labels are well-formed, since a typo would silently register as
a retrieval miss forever.
"""

from aws_mlops_support_agent.evals.dataset import EVAL_CASES

KNOWN_SOURCES = ("codebuild", "codepipeline")


def test_dataset_is_not_empty():
    assert len(EVAL_CASES) >= 15


def test_labels_point_at_real_corpus_sources():
    for case in EVAL_CASES:
        for f in case.expected_files:
            source_id, _, filename = f.partition("/")
            assert source_id in KNOWN_SOURCES, f
            assert filename.endswith(".md"), f


def test_escalation_expectation_matches_whether_labels_exist():
    """Off-corpus cases have no expected files; on-corpus cases have some."""
    for case in EVAL_CASES:
        assert (case.expected_files == ()) == case.should_escalate


def test_dataset_has_both_on_and_off_corpus_cases():
    """A set of only answerable questions can't catch a permissive threshold."""
    on_corpus = [c for c in EVAL_CASES if not c.should_escalate]
    off_corpus = [c for c in EVAL_CASES if c.should_escalate]
    assert len(on_corpus) >= 10
    assert len(off_corpus) >= 3


def test_questions_are_unique():
    questions = [c.question for c in EVAL_CASES]
    assert len(questions) == len(set(questions))


def test_every_case_explains_itself():
    assert all(c.notes for c in EVAL_CASES)

"""Eval runner: retrieval hit@k + escalation accuracy.

Measures the two cheap-but-decisive stages of a RAG system WITHOUT any LLM
calls (a run costs one embedding request per question):

- **Hit@k**: for on-corpus questions, did any expected doc file land in the
  top-k retrieved chunks? If retrieval misses, answer quality is already
  doomed, so this stage is measured in isolation.
- **Escalation accuracy**: does the caller's REAL routing decision send each
  question where it should? The router is injected rather than imported, so
  this runner never depends on any particular agent — that keeps it generic
  while still testing the real code path.

Each row also reports `top_score` and `score_gap`, so "does score_gap add
signal?" is answerable by eye from the saved table.

The dataset lives with the PROJECT (the questions are corpus-specific); only
the `EvalCase` type and this runner are generic.
"""

from collections.abc import Callable
from dataclasses import dataclass

from rag_core.retrieval.confidence import assess_confidence
from rag_core.retrieval.retriever import RetrievedChunk


@dataclass(frozen=True)
class EvalCase:
    question: str
    # "source_id/filename" entries that count as relevant; empty = off-corpus.
    expected_files: tuple[str, ...]
    # Should the system escalate (hand off to a human) instead of answering?
    should_escalate: bool
    # Why these files / why off-corpus.
    notes: str = ""


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    top_score: float
    score_gap: float
    escalated: bool  # what the router decided
    hit: bool | None  # None for off-corpus cases (no expected files)
    retrieved: tuple[str, ...]  # "source_id/filename" of top-k chunks, deduped

    @property
    def escalation_correct(self) -> bool:
        return self.escalated == self.case.should_escalate

    @property
    def passed(self) -> bool:
        """On-corpus: must hit AND route correctly. Off-corpus: route correctly."""
        return self.escalation_correct and (self.hit is None or self.hit)


@dataclass(frozen=True)
class Summary:
    on_corpus_total: int
    on_corpus_hits: int
    escalation_total: int
    escalation_correct: int


def chunk_label(chunk: RetrievedChunk) -> str:
    """How a retrieved chunk is matched against a case's expected files.

    File-level, not chunk-level: chunk labels would break every time chunking
    parameters change, while filenames survive re-ingestion.
    """
    return f"{chunk.source_id}/{chunk.source_file}"


def evaluate_case(
    case: EvalCase,
    retriever: Callable,
    k: int,
    min_top_score: float,
    should_escalate: Callable[[object], bool],
) -> CaseResult:
    """Run one question through real retrieval + the caller's real routing."""
    chunks = retriever(case.question, k=k)
    confidence = assess_confidence(chunks, min_top_score=min_top_score)
    escalated = should_escalate(confidence)

    # dict.fromkeys = order-preserving dedup (several chunks often come from
    # the same file).
    retrieved = tuple(dict.fromkeys(chunk_label(c) for c in chunks))
    hit = any(f in case.expected_files for f in retrieved) if case.expected_files else None

    return CaseResult(
        case=case,
        top_score=confidence.top_score,
        score_gap=confidence.score_gap,
        escalated=escalated,
        hit=hit,
        retrieved=retrieved,
    )


def run_cases(
    cases: list[EvalCase],
    retriever: Callable,
    k: int,
    min_top_score: float,
    should_escalate: Callable[[object], bool],
) -> list[CaseResult]:
    return [evaluate_case(c, retriever, k, min_top_score, should_escalate) for c in cases]


def summarize(results: list[CaseResult]) -> Summary:
    on_corpus = [r for r in results if r.hit is not None]
    return Summary(
        on_corpus_total=len(on_corpus),
        on_corpus_hits=sum(r.hit for r in on_corpus),
        escalation_total=len(results),
        escalation_correct=sum(r.escalation_correct for r in results),
    )


def format_results_table(
    results: list[CaseResult],
    summary: Summary,
    k: int,
    min_top_score: float,
) -> str:
    """Markdown table + summary, ready to paste into a README."""
    lines = [
        f"Retrieval eval — hit@{k} against the live index, escalation decided by the "
        f"agent's own router (threshold: top cosine < {min_top_score} → escalate).",
        "",
        f"| # | Question | Expected doc(s) | Hit@{k} | Top score | Gap | Escalated (want) | OK |",
        "|---|----------|-----------------|-------|-----------|-----|------------------|----|",
    ]
    for i, r in enumerate(results, 1):
        expected = ", ".join(f.split("/")[-1] for f in r.case.expected_files) or "—"
        hit = "—" if r.hit is None else ("yes" if r.hit else "**no**")
        want = "yes" if r.case.should_escalate else "no"
        escalated = f"{'yes' if r.escalated else 'no'} ({want})"
        ok = "✅" if r.passed else "❌"
        lines.append(
            f"| {i} | {r.case.question} | {expected} | {hit} "
            f"| {r.top_score:.3f} | {r.score_gap:.3f} | {escalated} | {ok} |"
        )

    lines += [
        "",
        f"**Hit@{k} (on-corpus):** {summary.on_corpus_hits}/{summary.on_corpus_total}",
        f"**Escalation accuracy:** {summary.escalation_correct}/{summary.escalation_total}",
    ]

    # Failed cases get their actual retrieved files listed — that's the
    # debugging signal (wrong file ranked higher? threshold off?).
    failures = [(i, r) for i, r in enumerate(results, 1) if not r.passed]
    if failures:
        lines += ["", "### Failures — what was actually retrieved", ""]
        for i, r in failures:
            lines.append(f"- **#{i}** {r.case.question}")
            for f in r.retrieved:
                lines.append(f"  - {f}")

    return "\n".join(lines)

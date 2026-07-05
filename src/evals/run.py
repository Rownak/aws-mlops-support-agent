"""Task 5.1 — Eval runner: retrieval hit@k + escalation accuracy.

Measures the two cheap-but-decisive stages of the agent WITHOUT any LLM
calls (the whole run costs ~15 embedding requests):

- **Hit@4**: for on-corpus questions, did any expected doc file land in the
  top-4 retrieved chunks? k=4 matches the agent's first retrieve attempt
  (`nodes.py`: k = 4 + 2*attempts). If retrieval misses, answer quality is
  already doomed, so we measure this stage in isolation.
- **Escalation accuracy**: does the agent's REAL routing function
  (`route_after_retrieve` from graph.py — not a re-implementation) send
  each question where it should? This is what actually judges the
  MIN_TOP_SCORE=0.35 threshold that task 2.3 deferred to Phase 5.

Each row also reports `top_score` and `score_gap` so the other deferred
question — "does score_gap add signal?" — is answerable by eye from the
saved table.

Run: `uv run python -m src.evals` — prints the markdown table and writes it
to `src/evals/results.md` (pasted into the README in task 6.5).
"""

from dataclasses import dataclass
from pathlib import Path

from src.agent.graph import route_after_retrieve
from src.evals.dataset import EVAL_CASES, EvalCase
from src.rag.confidence import MIN_TOP_SCORE, assess_confidence

# Same k as the agent's first retrieve attempt.
K = 4

RESULTS_PATH = Path(__file__).parent / "results.md"


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    top_score: float
    score_gap: float
    escalated: bool  # what the agent's router decided
    hit: bool | None  # None for off-corpus cases (no expected files)
    retrieved: tuple[str, ...]  # "service/filename" of top-k chunks, deduped

    @property
    def escalation_correct(self) -> bool:
        return self.escalated == self.case.should_escalate

    @property
    def passed(self) -> bool:
        """On-corpus: must hit AND route correctly. Off-corpus: route correctly."""
        return self.escalation_correct and (self.hit is None or self.hit)


def evaluate_case(case: EvalCase, retriever, k: int = K) -> CaseResult:
    """Run one question through real retrieval + the agent's real routing."""
    chunks = retriever(case.question, k=k)
    confidence = assess_confidence(chunks)
    # route_after_retrieve only reads state["confidence"], so a minimal
    # dict stands in for the full AgentState.
    escalated = route_after_retrieve({"confidence": confidence}) == "escalate"

    # dict.fromkeys = order-preserving dedup (several chunks often come
    # from the same file).
    retrieved = tuple(dict.fromkeys(f"{c.service}/{c.source_file}" for c in chunks))
    hit = any(f in case.expected_files for f in retrieved) if case.expected_files else None

    return CaseResult(
        case=case,
        top_score=confidence.top_score,
        score_gap=confidence.score_gap,
        escalated=escalated,
        hit=hit,
        retrieved=retrieved,
    )


@dataclass(frozen=True)
class Summary:
    on_corpus_total: int
    on_corpus_hits: int
    escalation_total: int
    escalation_correct: int


def summarize(results: list[CaseResult]) -> Summary:
    on_corpus = [r for r in results if r.hit is not None]
    return Summary(
        on_corpus_total=len(on_corpus),
        on_corpus_hits=sum(r.hit for r in on_corpus),
        escalation_total=len(results),
        escalation_correct=sum(r.escalation_correct for r in results),
    )


def format_results_table(results: list[CaseResult], summary: Summary) -> str:
    """Markdown table + summary, ready for the README."""
    lines = [
        f"Retrieval eval — hit@{K} against the live Pinecone index, escalation "
        f"decided by `route_after_retrieve` (threshold: top cosine < {MIN_TOP_SCORE} "
        "→ escalate).",
        "",
        "| # | Question | Expected doc(s) | Hit@4 | Top score | Gap | Escalated (want) | OK |",
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
        f"**Hit@{K} (on-corpus):** {summary.on_corpus_hits}/{summary.on_corpus_total}",
        f"**Escalation accuracy:** {summary.escalation_correct}/{summary.escalation_total}",
    ]

    # Failed cases get their actual retrieved files listed — that's the
    # debugging signal (wrong corpus file ranked higher? threshold off?).
    failures = [(i, r) for i, r in enumerate(results, 1) if not r.passed]
    if failures:
        lines += ["", "### Failures — what was actually retrieved", ""]
        for i, r in failures:
            lines.append(f"- **#{i}** {r.case.question}")
            for f in r.retrieved:
                lines.append(f"  - {f}")

    return "\n".join(lines)


def main() -> None:
    # Imported here so `pytest` can import this module without keys/network.
    import sys

    from src.config import load_config
    from src.rag.retriever import make_retriever

    # Windows consoles default to cp1252, which can't print the table's
    # arrows/check marks; the file write below is already explicit utf-8.
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_config()
    retriever = make_retriever(cfg)

    results = [evaluate_case(case, retriever) for case in EVAL_CASES]
    table = format_results_table(results, summarize(results))

    print(table)
    RESULTS_PATH.write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved to {RESULTS_PATH}")

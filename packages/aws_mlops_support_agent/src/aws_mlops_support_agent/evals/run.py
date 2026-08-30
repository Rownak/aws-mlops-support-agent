"""Run this project's eval set against the live index.

The generic machinery (hit@k, escalation accuracy, the markdown table) lives
in `rag_core.evals.runner`. What this module supplies is the corpus-specific
half: the questions, and the agent's REAL routing function — so the eval
measures the code that actually ships, not a re-implementation of it.

Run: `uv run aws-agent-evals` — prints the markdown table and writes it to
`evals/results.md`. Costs ~15 embedding calls and zero LLM calls.
"""

import sys
from pathlib import Path

from rag_core.evals.runner import format_results_table, run_cases, summarize
from rag_core.retrieval.retriever import make_retriever

from aws_mlops_support_agent.agent.graph import route_after_retrieve
from aws_mlops_support_agent.evals.dataset import EVAL_CASES
from aws_mlops_support_agent.settings import load_settings

RESULTS_PATH = Path(__file__).parent / "results.md"


def escalates(confidence) -> bool:
    """Ask the agent's own conditional edge where this retrieval would go.

    route_after_retrieve only reads state["confidence"], so a minimal dict
    stands in for the full AgentState.
    """
    return route_after_retrieve({"confidence": confidence}) == "escalate"


def main() -> None:
    # Windows consoles default to cp1252, which can't print the table's
    # arrows/check marks; the file write below is already explicit utf-8.
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_settings()
    retriever = make_retriever(cfg.rag)

    # Same k as the agent's first retrieve attempt (nodes.py: k = top_k + 2*attempts).
    k = cfg.rag.retrieval.top_k
    min_top_score = cfg.rag.retrieval.min_top_score

    results = run_cases(EVAL_CASES, retriever, k, min_top_score, escalates)
    table = format_results_table(results, summarize(results), k, min_top_score)

    print(table)
    RESULTS_PATH.write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

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

from langchain_core.documents import Document
from rag_core import RagCore
from rag_core.evals.runner import format_results_table, run_cases, summarize

from aws_mlops_support_agent.agent.graph import route_after_retrieve
from aws_mlops_support_agent.evals.dataset import EVAL_CASES
from aws_mlops_support_agent.settings import CONFIG_PATH

RESULTS_PATH = Path(__file__).parent / "results.md"


def escalates(confidence) -> bool:
    """Ask the agent's own conditional edge where this retrieval would go.

    route_after_retrieve only reads state["confidence"], so a minimal dict
    stands in for the full AgentState.
    """
    return route_after_retrieve({"confidence": confidence}) == "escalate"


def _basename_retriever(rag: RagCore):
    """Wrap rag.retrieve_scored so the runner's chunk_label sees a filename.

    rag_core.evals.runner.chunk_label() matches on the raw
    document.metadata["source"], which is the full local file path (and
    OS-specific: backslashes on Windows, forward slashes elsewhere) —
    matching EVAL_CASES.expected_files against that directly would be
    unportable. Rewriting `source` to just the filename here, rather than
    changing expected_files to full paths, keeps the dataset both portable
    and readable.
    """

    def _with_basename_source(doc: Document) -> Document:
        metadata = {**doc.metadata, "source": Path(doc.metadata["source"]).name}
        return Document(page_content=doc.page_content, metadata=metadata)

    def retriever(question: str, k: int) -> list[tuple[Document, float]]:
        scored = rag.retrieve_scored(question, k=k)
        return [(_with_basename_source(doc), score) for doc, score in scored]

    return retriever


def main() -> None:
    # Windows consoles default to cp1252, which can't print the table's
    # arrows/check marks; the file write below is already explicit utf-8.
    sys.stdout.reconfigure(encoding="utf-8")

    rag = RagCore(str(CONFIG_PATH))
    retriever = _basename_retriever(rag)

    # Same k as the agent's first retrieve attempt (nodes.py: k = top_k + 2*attempts).
    k = rag.config.retriever.top_k
    min_top_score = rag.config.retriever.min_top_score

    results = run_cases(EVAL_CASES, retriever, k, min_top_score, escalates)
    table = format_results_table(results, summarize(results), k, min_top_score)

    print(table)
    RESULTS_PATH.write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

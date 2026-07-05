"""Phase 2 sanity CLI — run the full retrieve -> confidence -> answer path.

Usage:
  uv run python -m src.agent.ask "How do I set environment variables in CodeBuild?"
  uv run python -m src.agent.ask "..." -k 4

This is the eyeball-it tool for Phase 2, like src.ingest.sanity_check was for
Phase 1. Phase 3 wires these same three functions into LangGraph nodes.
"""

import argparse

from src.agent.answer import generate_answer
from src.agent.confidence import assess_confidence
from src.agent.retriever import make_retriever
from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the doc corpus a question.")
    parser.add_argument("question", help="AWS CodeBuild/CodePipeline question")
    parser.add_argument("-k", type=int, default=4, help="Number of chunks to retrieve")
    args = parser.parse_args()

    cfg = load_config()
    retriever = make_retriever(cfg)

    chunks = retriever(args.question, k=args.k)
    print(f"\n=== Retrieved {len(chunks)} chunks ===")
    for i, chunk in enumerate(chunks, start=1):
        print(f"[{i}] score={chunk.score:.4f}  [{chunk.service}] {chunk.heading}")

    confidence = assess_confidence(chunks)
    verdict = "CONFIDENT" if confidence.is_confident else "LOW CONFIDENCE"
    print(f"\n=== Confidence: {verdict} ===")
    print(f"    {confidence.reason} (gap={confidence.score_gap:.3f})")

    # Still answer even on low confidence — the system prompt makes the model
    # admit gaps, and seeing that output is useful for tuning. Phase 3 is
    # where low confidence starts routing to escalation instead.
    print("\n=== Answer ===")
    print(generate_answer(args.question, chunks, cfg))


if __name__ == "__main__":
    main()

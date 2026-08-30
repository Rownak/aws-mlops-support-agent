"""Single-question CLI: run the full retrieve -> confidence -> answer path.

Generic on purpose — any project points it at its own config.yml:

  uv run rag-ask "How do I cache dependencies?" --config path/to/config.yml
  uv run rag-ask "..." --config path/to/config.yml -k 6

This is the eyeball-it tool for the query path. An agent wires these same
three functions into its own control flow.
"""

import argparse

from rag_core.config import load_config
from rag_core.generation.answer import generate_answer
from rag_core.retrieval.confidence import assess_confidence
from rag_core.retrieval.retriever import make_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a RAG corpus a question.")
    parser.add_argument("question", help="The question to answer from the corpus")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the project's config.yml (defaults to env vars only)",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=None,
        help="Number of chunks to retrieve (defaults to the configured top_k)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    retriever = make_retriever(cfg)

    chunks = retriever(args.question, k=args.k)
    print(f"\n=== Retrieved {len(chunks)} chunks ===")
    for i, chunk in enumerate(chunks, start=1):
        print(f"[{i}] score={chunk.score:.4f}  [{chunk.source_id}] {chunk.heading}")

    confidence = assess_confidence(chunks, min_top_score=cfg.retrieval.min_top_score)
    verdict = "CONFIDENT" if confidence.is_confident else "LOW CONFIDENCE"
    print(f"\n=== Confidence: {verdict} ===")
    print(f"    {confidence.reason} (gap={confidence.score_gap:.3f})")

    # Still answer even on low confidence — the system prompt makes the model
    # admit gaps, and seeing that output is useful for tuning. An agent is
    # where low confidence starts routing to escalation instead.
    print("\n=== Answer ===")
    print(generate_answer(args.question, chunks, cfg))


if __name__ == "__main__":
    main()

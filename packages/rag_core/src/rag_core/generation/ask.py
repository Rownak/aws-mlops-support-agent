"""Single-question CLI: run the full retrieve -> generate path.

Generic on purpose — any project points it at its own config.yaml:

  uv run rag-ask "How do I cache dependencies?" --config path/to/config.yaml
  uv run rag-ask "..." --config path/to/config.yaml -k 6

This is the eyeball-it tool for the query path. An agent wires the same
underlying functions (`sources`, `retriever`, `confidence`, `generation`)
into its own control flow instead of going through the RagCore facade.
"""

import argparse

from rag_core.pipeline import RagCore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a RAG corpus a question.")
    parser.add_argument("question", help="The question to answer from the corpus")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the project's config.yaml",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=None,
        help="Number of chunks to retrieve (defaults to the configured top_k)",
    )
    args = parser.parse_args()

    rag = RagCore(args.config)
    answer = rag.query(args.question, k=args.k)

    print(f"\n=== {'REFUSED' if answer.refused else 'ANSWER'} (confidence={answer.confidence:.2f}) ===")
    print(answer.formatted())


if __name__ == "__main__":
    main()

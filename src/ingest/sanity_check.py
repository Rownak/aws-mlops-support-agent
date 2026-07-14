"""Task 1.5 — Eyeball retrieval quality before building the agent.

Usage:
  uv run python -m src.ingest.sanity_check "How do I set environment variables in CodeBuild?"
  uv run python -m src.ingest.sanity_check "..." -k 3
"""

import argparse

from src.config import load_config
from src.ingest.index import get_vector_store_for_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Pinecone and print top-k chunks.")
    parser.add_argument("query", help="Question to search the doc corpus with")
    parser.add_argument("-k", type=int, default=5, help="Number of chunks to return")
    args = parser.parse_args()

    store = get_vector_store_for_query(load_config())
    # Returns (Document, score) pairs; cosine similarity, higher = closer.
    results = store.similarity_search_with_score(args.query, k=args.k)

    print(f"\nQuery: {args.query}\n")
    for rank, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata
        print(f"--- #{rank}  score={score:.4f}  [{meta.get('service')}] ---")
        print(f"    heading: {meta.get('heading')}")
        print(f"    file:    {meta.get('source_file')}")
        print(f"    url:     {meta.get('url')}")
        snippet = " ".join(doc.page_content.split())[:300]
        print(f"    text:    {snippet}\n")


if __name__ == "__main__":
    main()

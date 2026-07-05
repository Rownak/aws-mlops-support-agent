"""Ingestion pipeline entrypoint:  uv run python -m src.ingest

For every repo in sources.REPOS: fetch (clone + pre-archival checkout) ->
chunk (markdown -> Documents with metadata) -> index (embed + upsert).
Safe to re-run: clones are reused and upserts use deterministic IDs.
"""

from src.config import load_config
from src.ingest.chunk import chunk_repo
from src.ingest.fetch import fetch_repo
from src.ingest.index import upsert_chunks
from src.ingest.sources import REPOS


def main() -> None:
    cfg = load_config()

    all_chunks = []
    for repo in REPOS:
        doc_dir = fetch_repo(repo)
        all_chunks.extend(chunk_repo(repo, doc_dir))

    upsert_chunks(cfg, all_chunks)
    print(
        f"[done] {len(all_chunks)} chunks from {len(REPOS)} repos "
        f"in index '{cfg.pinecone_index_name}'"
    )


if __name__ == "__main__":
    main()

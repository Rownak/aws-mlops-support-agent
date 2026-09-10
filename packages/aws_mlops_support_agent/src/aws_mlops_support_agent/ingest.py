"""Ingestion entrypoint:  uv run aws-agent-ingest

One step: `RagCore.sync()`. The AWS-specific work — cloning each archived
awsdocs repo, recovering its pre-archival markdown, stripping anchor noise,
and tagging each file with its canonical docs URL — lives in this project's
`awsdocs_git` source type, which rag_core drives like any other source.

Safe to re-run: clones are reused and upserts use deterministic IDs.

``--save-chunks-local`` additionally dumps each document's chunks to
``data/aws_docs/chunks/{source_id}/{doc_stem}/`` as JSON, for inspecting what
got embedded without a Pinecone round-trip. Off by default; rag_core's
ingestion is unaffected unless this flag opts in.
"""

import argparse
import json
import re
from pathlib import Path

from langchain_core.documents import Document

from rag_core import RagCore
from rag_core.observability import setup_json_logging

# Importing the sources package registers `awsdocs_git` with rag_core's
# REGISTRY, which is what makes config.yml's `type:` resolvable below.
import aws_mlops_support_agent.sources  # noqa: F401
from aws_mlops_support_agent.settings import CONFIG_PATH

CHUNKS_DIR = Path("data/aws_docs/chunks")

# Matches how AwsDocsGitSource lays out a clone:
# data/aws_docs/{source_id}/doc_source/{doc_stem}.md
_DOC_SOURCE_PATH = re.compile(r"[/\\]([^/\\]+)[/\\]doc_source[/\\]([^/\\]+)\.md$")


def _save_chunks_local(file_path: str, chunks: list[Document]) -> None:
    """Dump one document's chunks to data/aws_docs/chunks/{source_id}/{doc_stem}/."""
    match = _DOC_SOURCE_PATH.search(file_path)
    if match is None:
        # Not an awsdocs_git-shaped path (e.g. a future non-git source) --
        # fall back to a flat directory rather than silently dropping chunks.
        source_id, doc_stem = "_unknown", Path(file_path).stem
    else:
        source_id, doc_stem = match.group(1), match.group(2)

    out_dir = CHUNKS_DIR / source_id / doc_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for chunk in chunks:
        index = chunk.metadata.get("chunk_index", 0)
        out_path = out_dir / f"chunk_{index:04d}.json"
        out_path.write_text(
            json.dumps(
                {"text": chunk.page_content, "metadata": chunk.metadata},
                indent=2,
            ),
            encoding="utf-8",
        )


def main() -> None:
    setup_json_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-chunks-local",
        action="store_true",
        help=f"Also dump each document's chunks as JSON under {CHUNKS_DIR}/",
    )
    args = parser.parse_args()

    on_chunks_prepared = _save_chunks_local if args.save_chunks_local else None
    rag = RagCore(str(CONFIG_PATH), on_chunks_prepared=on_chunks_prepared)
    stats = rag.sync()
    print(
        f"Ingestion complete: {stats['processed']}/{stats['total']} documents "
        f"({stats['skipped']} skipped, {stats['failed']} failed, "
        f"{stats['replaced']} replaced, {stats['chunks_created']} chunks)"
    )
    if stats["errors"]:
        print("Errors:")
        for err in stats["errors"]:
            print(f"  {err['file']}: {err['error']}")


if __name__ == "__main__":
    main()

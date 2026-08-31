"""Ingestion entrypoint:  uv run aws-agent-ingest

Two steps: first recover the AWS docs (clone, checkout pre-archival history,
strip anchors) into the local folders config.yml's `sources` point at — that
part is entirely AWS-specific and lives in `sources.fetch`. Then hand off to
rag_core's `RagCore.sync()`, which is corpus-agnostic: it just walks those
folders and does chunk/embed/upsert.

Safe to re-run: clones are reused and upserts use deterministic IDs.
"""

from rag_core import RagCore
from rag_core.observability import setup_json_logging

from aws_mlops_support_agent.settings import CONFIG_PATH, load_settings
from aws_mlops_support_agent.sources.fetch import fetch_all


def main() -> None:
    setup_json_logging()
    cfg = load_settings()

    # AWS-specific: populate the local folders config.yml's `sources` point
    # at. rag_core never learns what an awsdocs repo is.
    fetch_all(cfg.rag.sources)

    rag = RagCore(str(CONFIG_PATH))
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

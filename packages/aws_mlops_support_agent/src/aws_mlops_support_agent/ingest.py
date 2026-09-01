"""Ingestion entrypoint:  uv run aws-agent-ingest

One step: `RagCore.sync()`. The AWS-specific work — cloning each archived
awsdocs repo, recovering its pre-archival markdown, stripping anchor noise,
and tagging each file with its canonical docs URL — lives in this project's
`awsdocs_git` source type, which rag_core drives like any other source.

Safe to re-run: clones are reused and upserts use deterministic IDs.
"""

from rag_core import RagCore
from rag_core.observability import setup_json_logging

# Importing the sources package registers `awsdocs_git` with rag_core's
# REGISTRY, which is what makes config.yml's `type:` resolvable below.
import aws_mlops_support_agent.sources  # noqa: F401
from aws_mlops_support_agent.settings import CONFIG_PATH


def main() -> None:
    setup_json_logging()

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

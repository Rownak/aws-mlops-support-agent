"""
What one ingestion run did, and the shape every ingest verb reports back.

Split out from the ingestor itself because these are the run's *result*
type, not its machinery: callers (CLIs, eval scripts, the agent) read
`IngestStats` without caring how the writing happened, and the ingestor
is easier to read when the bookkeeping types are not in the way.
"""

from typing import TypedDict


class IngestError(TypedDict):
    file: str
    error: str


class IngestStats(TypedDict):
    """What one ingestion run did."""

    total: int
    processed: int
    skipped: int
    failed: int
    chunks_created: int
    #: Documents whose content changed since a previous ingest, where the
    #: older version's chunks were removed before writing the new one.
    replaced: int
    errors: list[IngestError]


def empty_stats(total: int = 0) -> IngestStats:
    """A zeroed IngestStats, optionally with the run's document count set."""
    return {
        "total": total,
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "chunks_created": 0,
        "replaced": 0,
        "errors": [],
    }

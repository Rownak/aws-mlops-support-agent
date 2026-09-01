"""Writing documents into the vector store: chunking, dedup, replacement."""

from .ingestor import DEFAULT_BATCH_SIZE, Ingestor
from .stats import IngestError, IngestStats, empty_stats

__all__ = [
    "Ingestor",
    "DEFAULT_BATCH_SIZE",
    "IngestStats",
    "IngestError",
    "empty_stats",
]

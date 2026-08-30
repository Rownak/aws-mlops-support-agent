"""The extension seam: what a corpus adapter must provide (design.md §6).

`rag_core` cannot know how to fetch every corpus — awsdocs needs git-history
recovery, a dataset needs an HTTP download, an internal corpus needs S3. So
the engine defines this narrow protocol and each project implements it.

Dependencies point ONE way only: a project imports `rag_core`, never the
reverse. `rag_core.pipeline` consumes `DocSource` and never learns what an
AWS doc repo is.

Implementing a source means returning `LoadedDoc`s — raw text plus enough
provenance to build a citation. Everything after that (chunking, embedding,
upserting) is the engine's job.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from rag_core.config import SourceSpec


@dataclass(frozen=True)
class LoadedDoc:
    """One raw document, before chunking.

    `source_id` is the neutral name for what used to be called "service" — it
    identifies which configured source this document came from, and ends up
    in every chunk's metadata and chunk ID.
    """

    source_id: str
    # Path or filename within the source, e.g. "build-caching.md". Used for
    # deterministic chunk IDs and as the eval set's relevance label.
    source_file: str
    # The document's raw text (markdown today; a loader may add other formats).
    text: str
    # Public URL for citations. The adapter builds it — only the adapter knows
    # how a local filename maps onto a published page.
    url: str = ""
    # Anything else worth carrying into chunk metadata.
    extra: dict = field(default_factory=dict)


@runtime_checkable
class DocSource(Protocol):
    """A corpus adapter: knows how to obtain the documents of ONE source.

    Runtime-checkable so a project (or a test) can assert its adapter really
    satisfies the seam with `isinstance(adapter, DocSource)`.
    """

    # Which configured source this adapter serves. The pipeline reads
    # `spec.id` for logging and never inspects `spec.options` itself.
    spec: SourceSpec

    def fetch(self) -> Iterable[LoadedDoc]:
        """Obtain the documents, doing whatever I/O this corpus requires.

        Returning an iterable (rather than a list) lets a large corpus stream
        instead of materializing every document in memory at once.
        """
        ...


def build_sources(specs: Iterable[SourceSpec], registry: dict) -> Iterator[DocSource]:
    """Turn configured specs into adapters using a project's loader registry.

    `registry` maps a `loader` name from config.yml (e.g. "awsdocs_git") to a
    callable taking the SourceSpec and returning a DocSource. The registry is
    supplied by the PROJECT — that is what keeps corpus knowledge out of here.
    """
    for spec in specs:
        factory = registry.get(spec.loader)
        if factory is None:
            known = ", ".join(sorted(registry)) or "(none registered)"
            raise RuntimeError(
                f"Source '{spec.id}' asks for loader '{spec.loader}', which is not "
                f"registered. Known loaders: {known}."
            )
        yield factory(spec)

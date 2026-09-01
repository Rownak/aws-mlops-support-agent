"""
Turning one document's text into the chunks that get embedded and stored.

Sits between `splitter` (how text is cut) and the pipeline (what is written):
`splitter_from_config` picks the splitter a config asks for, and
`build_chunks` runs it and stamps each piece with the provenance metadata
retrieval and deduplication later depend on.

Both are plain functions with no vector store and no config-file access, so
chunking output can be inspected and tested on its own — no Pinecone
connection, no `RagCore` instance.
"""

from langchain_core.documents import Document

from .hashing import sha256_chunk, sha256_text
from .splitter import get_markdown_splitter, get_splitter

#: Keys rag_core itself stamps onto every chunk. A source's extra metadata
#: must not use any of these — `build_chunks` raises rather than let one be
#: overwritten, because dedup and partial-ingest detection read them back.
#: Keep in sync with the metadata dict `build_chunks` builds below.
RESERVED_METADATA_KEYS = frozenset(
    {
        "source",
        "file_name",
        "file_type",
        "file_hash",
        "chunk_id",
        "chunk_hash",
        "content_hash",
        "chunk_index",
        "total_chunks",
    }
)


def splitter_from_config(splitter_cfg):
    """
    Build the splitter a `splitter` config block asks for.

    Args:
        splitter_cfg: The config's `splitter` block (`strategy`,
            `chunk_size`, `chunk_overlap`)

    Returns:
        A LangChain text splitter, ready to pass to `build_chunks`
    """
    if splitter_cfg.strategy == "markdown":
        return get_markdown_splitter(splitter_cfg.chunk_size, splitter_cfg.chunk_overlap)
    return get_splitter(splitter_cfg.chunk_size, splitter_cfg.chunk_overlap)


def build_chunks(
    text: str,
    file_path: str,
    file_name: str,
    file_type: str,
    file_hash: str,
    splitter,
    extra_metadata: dict | None = None,
) -> list[Document]:
    """
    Split one document's text into chunks carrying provenance metadata.

    ``total_chunks`` is what makes a partial ingest detectable later: a run
    that died mid-write leaves fewer stored chunks than each chunk claims.

    Args:
        text: The document's extracted text
        file_path: Path the document was read from, kept as ``source``
        file_name: The document's filename
        file_type: The document's extension, without the dot
        file_hash: SHA256 of the file's contents, shared by all its chunks
        splitter: The splitter to cut the text with
        extra_metadata: Extra metadata merged into every chunk of this file —
            typically what a `Source.metadata_for()` returned, e.g. a
            canonical URL. Must not use a key in `RESERVED_METADATA_KEYS`.

    Returns:
        One Document per piece, in order. Empty when the text yields no
        pieces — the caller decides whether that is an error.

    Raises:
        ValueError: If `extra_metadata` uses a reserved key. Raised rather
            than silently overwriting, because the reserved keys drive dedup
            and partial-ingest detection.

    Example:
        >>> chunks = build_chunks(
        ...     "a\\n\\nb", "docs/a.md", "a.md", "md", "abc123",
        ...     get_markdown_splitter(800, 100),
        ... )  # doctest: +SKIP
    """
    # Checked once per file, not per chunk: extra_metadata is the same
    # mapping for every piece this document produces.
    if extra_metadata:
        collisions = set(extra_metadata) & RESERVED_METADATA_KEYS
        if collisions:
            raise ValueError(
                f"{file_path}: extra metadata uses reserved key(s) "
                f"{sorted(collisions)}; reserved: {sorted(RESERVED_METADATA_KEYS)}"
            )

    pieces = splitter.split_text(text)
    documents = []

    for i, piece in enumerate(pieces):
        chunk_id = f"{file_hash}_{i}"
        metadata = {
            "source": file_path,
            "file_name": file_name,
            "file_type": file_type,
            "file_hash": file_hash,
            "chunk_id": chunk_id,
            # Mixes in chunk_id, so identical text in two chunks
            # still hashes differently.
            "chunk_hash": sha256_chunk(chunk_id, piece),
            # Text alone — comparable across chunks and documents.
            "content_hash": sha256_text(piece),
            "chunk_index": i,
            "total_chunks": len(pieces),
        }
        # Safe as an unconditional merge: the guard above already rejected
        # any overlap with the reserved keys.
        if extra_metadata:
            metadata.update(extra_metadata)
        documents.append(Document(page_content=piece, metadata=metadata))
    return documents

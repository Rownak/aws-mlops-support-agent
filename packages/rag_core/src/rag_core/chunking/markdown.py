"""Structure-aware markdown chunking with citation metadata.

Two-stage split:
  1. MarkdownHeaderTextSplitter cuts along #/##/### headings, so chunks follow
     the document's own structure and we know which section each chunk came
     from (needed for citations and, in the agent, the "docs checked" field).
  2. RecursiveCharacterTextSplitter (token-based) further splits sections that
     are still too big. Sizes are measured in TOKENS via tiktoken — the unit
     the embedding model actually sees — not characters.

Everything corpus-specific is configuration: chunk size, overlap, and the
`strip_patterns` regexes applied before splitting. The awsdocs `<a name>`
heading anchor is one configured pattern, not a constant in the engine.
"""

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from rag_core.config import ChunkingConfig
from rag_core.sources import LoadedDoc

# strip_headers=False keeps the heading text inside the chunk body — headings
# carry strong keywords, which helps both embedding quality and readability.
_header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=False,
)


def _size_splitter(cfg: ChunkingConfig) -> RecursiveCharacterTextSplitter:
    """Build the token-based splitter for this config.

    cl100k_base is the tokenizer used by the text-embedding-3-* models.
    """
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=cfg.size_tokens,
        chunk_overlap=cfg.overlap_tokens,
    )


def strip_patterns(text: str, patterns: tuple[str, ...]) -> str:
    """Remove each configured regex from the raw text before splitting."""
    for pattern in patterns:
        text = re.sub(pattern, "", text)
    return text


def chunk_document(doc: LoadedDoc, cfg: ChunkingConfig) -> list[Document]:
    """Split one loaded document into chunks with metadata + deterministic IDs."""
    size_splitter = _size_splitter(cfg)
    stem = Path(doc.source_file).stem
    chunks: list[Document] = []

    for section in _header_splitter.split_text(strip_patterns(doc.text, cfg.strip_patterns)):
        # The header splitter stores matched headings in metadata as h1/h2/h3.
        heading = " > ".join(
            section.metadata[level] for level in ("h1", "h2", "h3") if level in section.metadata
        )
        for piece in size_splitter.split_text(section.page_content):
            chunks.append(
                Document(
                    # Deterministic ID: re-ingesting overwrites the same vectors
                    # instead of duplicating them (idempotent upserts).
                    id=f"{doc.source_id}/{stem}#{len(chunks)}",
                    page_content=piece,
                    metadata={
                        # Neutral name for what an AWS-only codebase called
                        # "service" — a corpus of medical records has sources
                        # too, but no services.
                        "source_id": doc.source_id,
                        "source_file": doc.source_file,
                        "heading": heading,
                        "url": doc.url,
                        **doc.extra,
                    },
                )
            )
    return chunks


def chunk_documents(docs, cfg: ChunkingConfig) -> list[Document]:
    """Chunk every loaded document from a source."""
    chunks: list[Document] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, cfg))
    return chunks

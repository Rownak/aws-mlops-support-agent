"""Tasks 1.2 + 1.3 — Structure-aware chunking with citation metadata.

Two-stage split:
  1. MarkdownHeaderTextSplitter cuts along #/##/### headings, so chunks
     follow the document's own structure and we know which section each
     chunk came from (needed for citations and the Jira "docs checked" field).
  2. RecursiveCharacterTextSplitter (token-based) further splits sections
     that are still too big. Sizes are measured in TOKENS via tiktoken —
     the unit the embedding model actually sees — not characters.
"""

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.ingest.sources import DocRepo

# ~800 tokens sits mid-range of the 500-1000 target; 100 tokens ~ 12% overlap
# so a sentence cut at a boundary still appears whole in one of the chunks.
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 100

# strip_headers=False keeps the heading text inside the chunk body — headings
# carry strong keywords, which helps both embedding quality and readability.
_header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=False,
)

# awsdocs markdown embeds HTML anchors in every heading, e.g.
# "# Build environments<a name="build-env"></a>". Strip them so headings are
# clean in citations / Jira fields and don't add noise to the embeddings.
_ANCHOR_RE = re.compile(r'<a name="[^"]*"></a>')

# cl100k_base is the tokenizer used by the text-embedding-3-* models.
_size_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=CHUNK_SIZE_TOKENS,
    chunk_overlap=CHUNK_OVERLAP_TOKENS,
)


def build_doc_url(repo: DocRepo, source_file: str) -> str:
    """doc_source/foo.md -> https://docs.aws.amazon.com/.../foo.html"""
    return f"{repo.docs_base_url}{Path(source_file).stem}.html"


def chunk_markdown(repo: DocRepo, source_file: str, text: str) -> list[Document]:
    """Split one markdown document into chunks with metadata + deterministic IDs."""
    chunks: list[Document] = []
    for section in _header_splitter.split_text(_ANCHOR_RE.sub("", text)):
        # The header splitter stores matched headings in metadata as h1/h2/h3.
        heading = " > ".join(
            section.metadata[level] for level in ("h1", "h2", "h3") if level in section.metadata
        )
        for piece in _size_splitter.split_text(section.page_content):
            chunks.append(
                Document(
                    # Deterministic ID: re-ingesting overwrites the same Pinecone
                    # vectors instead of duplicating them (idempotent upserts).
                    id=f"{repo.service}/{Path(source_file).stem}#{len(chunks)}",
                    page_content=piece,
                    metadata={
                        "service": repo.service,
                        "source_file": source_file,
                        "heading": heading,
                        "url": build_doc_url(repo, source_file),
                    },
                )
            )
    return chunks


def chunk_repo(repo: DocRepo, doc_dir: Path) -> list[Document]:
    """Chunk every markdown file in a repo's doc_source directory."""
    docs: list[Document] = []
    for md_file in sorted(doc_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        docs.extend(chunk_markdown(repo, md_file.name, text))
    print(f"[chunk] {repo.service}: {len(docs)} chunks")
    return docs

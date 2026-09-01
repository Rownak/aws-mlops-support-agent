"""
Text splitting utilities for RAG pipeline.

Provides configurable text splitters using RecursiveCharacterTextSplitter
from LangChain for chunking documents into appropriate sizes for
embedding and retrieval.

Sizes here are **tokens, not characters**. Tokens are what actually
constrains an embedding model's input, so a `chunk_size` of 800 means 800
tokens (very roughly 3-4x that many characters of English prose). Every
splitter below counts with the same tiktoken encoder, so one configured
number means the same thing whichever strategy a config picks.

Reference: https://docs.langchain.com/oss/python/integrations/splitters
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

#: The encoder chunk sizes are counted with. `cl100k_base` is not tied to
#: whichever embedding provider is configured — it is a stable, provider-
#: neutral yardstick, so changing embedding models does not silently
#: resize every chunk. Exact only for models that share this vocabulary;
#: elsewhere it is a close and, more importantly, *consistent* estimate.
TOKEN_ENCODING = "cl100k_base"

#: Fallback separator ladder: paragraphs, then lines, then words, then
#: characters. Each rung is a worse place to cut than the one before.
DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]

#: Markdown structure first, prose structure second. Header rungs come
#: before the paragraph rung so a section boundary is preferred over an
#: arbitrary blank line. `# ` (h1) is included so top-level sections are
#: break candidates too.
MARKDOWN_SEPARATORS = [
    "\n\n# ",  # Level 1 headers
    "\n\n## ",  # Level 2 headers
    "\n\n### ",  # Level 3 headers
    "\n\n#### ",  # Level 4 headers
    "\n\n",  # Paragraph breaks
    "\n",  # Line breaks
    " ",  # Words
    "",  # Characters
]

#: Code structure first, then prose fallbacks.
CODE_SEPARATORS = [
    "\n\nclass ",  # Class definitions
    "\n\ndef ",  # Function definitions
    "\n\n#",  # Comments
    "\n\n",  # Paragraph breaks
    "\n",  # Line breaks
    " ",  # Words
    "",  # Characters
]


def _token_splitter(
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> RecursiveCharacterTextSplitter:
    """
    Build a recursive splitter that measures chunks in tokens.

    `from_tiktoken_encoder` is what swaps the length function from
    `len` (characters) to a token count; everything else is the ordinary
    recursive splitter. Keeping the construction in one place is what
    guarantees all three strategies below count the same way.

    Args:
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Tokens shared between neighbouring chunks
        separators: Break points to try, best first

    Returns:
        A splitter whose `split_text` returns token-bounded pieces
    """
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TOKEN_ENCODING,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        # Separators are structural markers (headers, `def `), so dropping
        # them would delete the heading text itself from the chunk.
        keep_separator=True,
        strip_whitespace=True,
    )


def get_splitter(
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    separators: list[str] | None = None,
) -> RecursiveCharacterTextSplitter:
    """
    Get a general-purpose splitter, sized in tokens.

    Splits by trying each separator in turn until the pieces fit the
    token budget.

    Args:
        chunk_size: Maximum tokens per chunk (default: 800)
        chunk_overlap: Tokens to overlap between chunks (default: 100)
        separators: Separators to try in order; defaults to
            `DEFAULT_SEPARATORS`

    Returns:
        Configured RecursiveCharacterTextSplitter instance

    Example:
        >>> splitter = get_splitter(chunk_size=500, chunk_overlap=100)
        >>> chunks = splitter.split_text(long_document)  # doctest: +SKIP
    """
    return _token_splitter(
        chunk_size, chunk_overlap, separators or DEFAULT_SEPARATORS
    )


def get_markdown_splitter(
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> RecursiveCharacterTextSplitter:
    """
    Get a splitter tuned for markdown documents, sized in tokens.

    Prefers to break at header boundaries so a chunk tends to hold one
    section rather than the tail of one and the head of the next.

    Note this is header-*aware*, not a true header splitter: headings are
    preferred break points, but no `heading` metadata is extracted and a
    section longer than the budget is still cut mid-section.

    Args:
        chunk_size: Maximum tokens per chunk (default: 800)
        chunk_overlap: Tokens to overlap between chunks (default: 100)

    Returns:
        Markdown-optimized RecursiveCharacterTextSplitter

    Example:
        >>> splitter = get_markdown_splitter(chunk_size=2000)
        >>> chunks = splitter.split_text(markdown_content)  # doctest: +SKIP
    """
    return _token_splitter(chunk_size, chunk_overlap, MARKDOWN_SEPARATORS)


def get_code_splitter(
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> RecursiveCharacterTextSplitter:
    """
    Get a splitter tuned for code documents, sized in tokens.

    Prefers to break at class, function and comment boundaries so a chunk
    tends to hold whole definitions.

    Args:
        chunk_size: Maximum tokens per chunk (default: 800)
        chunk_overlap: Tokens to overlap between chunks (default: 100)

    Returns:
        Code-optimized RecursiveCharacterTextSplitter

    Example:
        >>> splitter = get_code_splitter(chunk_size=1000)
        >>> chunks = splitter.split_text(code_content)  # doctest: +SKIP
    """
    return _token_splitter(chunk_size, chunk_overlap, CODE_SEPARATORS)

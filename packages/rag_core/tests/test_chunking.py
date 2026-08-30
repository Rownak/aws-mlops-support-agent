"""Unit tests for markdown chunking. No network, no API keys.

The AWS anchor regex is supplied via ChunkingConfig here — proving the engine
carries no corpus-specific cleanup rule of its own.
"""

from rag_core.chunking.markdown import chunk_document, chunk_documents, strip_patterns
from rag_core.config import ChunkingConfig
from rag_core.sources import LoadedDoc

# The awsdocs heading-anchor pattern, passed in as configuration.
AWS_ANCHOR = '<a name="[^"]*"></a>'

CFG = ChunkingConfig(size_tokens=800, overlap_tokens=100, strip_patterns=(AWS_ANCHOR,))

SAMPLE_MD = """\
# Build environments

Intro paragraph about build environments.

## Environment variables

You can set environment variables in the buildspec file.

### Reserved variables

CODEBUILD_BUILD_ID is reserved.
"""


def make_doc(text: str, source_file: str = "build-env.md") -> LoadedDoc:
    return LoadedDoc(
        source_id="codebuild",
        source_file=source_file,
        text=text,
        url=f"https://docs.aws.amazon.com/codebuild/latest/userguide/"
        f"{source_file.removesuffix('.md')}.html",
    )


def test_chunks_carry_citation_metadata():
    chunks = chunk_document(make_doc(SAMPLE_MD), CFG)
    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        # Renamed from "service": the engine has no notion of AWS services.
        assert chunk.metadata["source_id"] == "codebuild"
        assert chunk.metadata["source_file"] == "build-env.md"
        assert chunk.metadata["url"].endswith("/build-env.html")
        assert chunk.page_content.strip()
    # Heading path reflects the markdown structure (h1 > h2 > h3 join).
    headings = [c.metadata["heading"] for c in chunks]
    assert "Build environments > Environment variables > Reserved variables" in headings


def test_chunk_ids_are_deterministic_and_unique():
    first = chunk_document(make_doc(SAMPLE_MD), CFG)
    second = chunk_document(make_doc(SAMPLE_MD), CFG)
    ids = [c.id for c in first]
    assert ids == [c.id for c in second]  # same input -> same IDs (idempotent upserts)
    assert len(ids) == len(set(ids))  # no collisions within a file
    assert ids[0] == "codebuild/build-env#0"


def test_configured_strip_pattern_removes_awsdocs_anchors():
    md = '# Build environments<a name="build-env-ref"></a>\n\nSome text.\n'
    chunks = chunk_document(make_doc(md), CFG)
    assert chunks[0].metadata["heading"] == "Build environments"
    assert "<a name=" not in chunks[0].page_content


def test_anchors_survive_when_no_pattern_is_configured():
    """Proves the cleanup is configuration, not behavior baked into the engine."""
    md = '# Build environments<a name="build-env-ref"></a>\n\nSome text.\n'
    chunks = chunk_document(make_doc(md), ChunkingConfig())
    assert "<a name=" in chunks[0].metadata["heading"]


def test_strip_patterns_applies_every_configured_regex():
    text = 'keep <!-- drop --> this <a name="x"></a>too'
    assert strip_patterns(text, (AWS_ANCHOR, "<!--.*?-->")) == "keep  this too"


def test_oversized_sections_are_split():
    # One section far beyond the token budget must yield multiple chunks.
    big = "# Big\n\n" + ("word " * CFG.size_tokens * 3)
    chunks = chunk_document(make_doc(big, "big.md"), CFG)
    assert len(chunks) > 1
    assert all(c.metadata["heading"] == "Big" for c in chunks)


def test_chunk_size_is_config_driven():
    """A smaller configured budget must produce more chunks from one input."""
    big = "# Big\n\n" + ("word " * 2000)
    doc = make_doc(big, "big.md")
    large = chunk_document(doc, ChunkingConfig(size_tokens=800, overlap_tokens=0))
    small = chunk_document(doc, ChunkingConfig(size_tokens=200, overlap_tokens=0))
    assert len(small) > len(large)


def test_extra_metadata_is_carried_through():
    doc = LoadedDoc(
        source_id="s", source_file="f.md", text="# H\n\ntext", url="u", extra={"lang": "en"}
    )
    assert chunk_document(doc, CFG)[0].metadata["lang"] == "en"


def test_chunk_documents_processes_every_document():
    docs = [make_doc(SAMPLE_MD, "a.md"), make_doc(SAMPLE_MD, "b.md")]
    files = {c.metadata["source_file"] for c in chunk_documents(docs, CFG)}
    assert files == {"a.md", "b.md"}

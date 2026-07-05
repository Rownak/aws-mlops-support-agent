"""Unit tests for the pure parts of ingestion: URL building and chunking.

No network, no API keys — fetch/index are verified by running the real
pipeline (see tasks.md testing philosophy).
"""

from src.ingest.chunk import CHUNK_SIZE_TOKENS, build_doc_url, chunk_markdown
from src.ingest.sources import DocRepo

REPO = DocRepo(
    service="codebuild",
    git_url="https://github.com/awsdocs/aws-codebuild-user-guide.git",
    docs_base_url="https://docs.aws.amazon.com/codebuild/latest/userguide/",
)

SAMPLE_MD = """\
# Build environments

Intro paragraph about build environments.

## Environment variables

You can set environment variables in the buildspec file.

### Reserved variables

CODEBUILD_BUILD_ID is reserved.
"""


def test_build_doc_url_maps_md_to_html():
    url = build_doc_url(REPO, "build-env-ref-env-vars.md")
    assert (
        url == "https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref-env-vars.html"
    )


def test_chunks_carry_citation_metadata():
    chunks = chunk_markdown(REPO, "build-env.md", SAMPLE_MD)
    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        assert chunk.metadata["service"] == "codebuild"
        assert chunk.metadata["source_file"] == "build-env.md"
        assert chunk.metadata["url"].endswith("/build-env.html")
        assert chunk.page_content.strip()
    # Heading path reflects the markdown structure (h1 > h2 > h3 join).
    headings = [c.metadata["heading"] for c in chunks]
    assert "Build environments > Environment variables > Reserved variables" in headings


def test_chunk_ids_are_deterministic_and_unique():
    first = chunk_markdown(REPO, "build-env.md", SAMPLE_MD)
    second = chunk_markdown(REPO, "build-env.md", SAMPLE_MD)
    ids = [c.id for c in first]
    assert ids == [c.id for c in second]  # same input -> same IDs (idempotent upserts)
    assert len(ids) == len(set(ids))  # no collisions within a file
    assert ids[0] == "codebuild/build-env#0"


def test_awsdocs_anchor_tags_are_stripped():
    md = '# Build environments<a name="build-env-ref"></a>\n\nSome text.\n'
    chunks = chunk_markdown(REPO, "build-env.md", md)
    assert chunks[0].metadata["heading"] == "Build environments"
    assert "<a name=" not in chunks[0].page_content


def test_oversized_sections_are_split():
    # One section far beyond the token budget must yield multiple chunks.
    big_section = "# Big\n\n" + ("word " * CHUNK_SIZE_TOKENS * 3)
    chunks = chunk_markdown(REPO, "big.md", big_section)
    assert len(chunks) > 1
    assert all(c.metadata["heading"] == "Big" for c in chunks)

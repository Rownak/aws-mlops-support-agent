"""Tests for the awsdocs_git rag_core source.

The git-history recovery itself is verified by running the real ingestion (it
needs actual repos); what is tested here is the seam: the source registers
under its config `type`, is configured entirely from config.yml, strips AWS's
anchor noise, and hands each file the docs URL that ends up on every chunk.
"""

import pytest

# Importing the package registers awsdocs_git with rag_core's REGISTRY.
import aws_mlops_support_agent.sources  # noqa: F401
from aws_mlops_support_agent.sources.fetch import AwsDocsGitSource, _strip_anchors
from rag_core.processing.chunking import RESERVED_METADATA_KEYS
from rag_core.sources import REGISTRY, Source, build_sources

CODEBUILD_URLS = {
    "git_url": "https://github.com/awsdocs/aws-codebuild-user-guide.git",
    "docs_base_url": "https://docs.aws.amazon.com/codebuild/latest/userguide/",
}
CODEPIPELINE_URLS = {
    "git_url": "https://github.com/awsdocs/aws-codepipeline-user-guide.git",
    "docs_base_url": "https://docs.aws.amazon.com/codepipeline/latest/userguide/",
}
SPEC = {"type": "awsdocs_git", "id": "codebuild", "path": "data/aws_docs", **CODEBUILD_URLS}


def test_source_is_registered_under_its_config_type():
    assert "awsdocs_git" in REGISTRY
    (source,) = build_sources([SPEC])
    assert isinstance(source, AwsDocsGitSource)
    assert isinstance(source, Source)


def test_source_is_configured_from_the_config_entry():
    source = AwsDocsGitSource(**{k: v for k, v in SPEC.items() if k != "type"})
    assert source.id == "codebuild"
    assert source.git_url.endswith("aws-codebuild-user-guide.git")
    # The clone and the docs inside it both hang off the configured path.
    assert source.repo_dir.parts[-2:] == ("aws_docs", "codebuild")
    assert source.doc_dir.name == "doc_source"


def test_doc_url_maps_md_to_html():
    url = AwsDocsGitSource(id="codebuild", **CODEBUILD_URLS).doc_url("build-env-ref-env-vars.md")
    assert (
        url == "https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref-env-vars.html"
    )


def test_metadata_for_supplies_the_url_that_rides_on_every_chunk():
    source = AwsDocsGitSource(id="codepipeline", **CODEPIPELINE_URLS)
    metadata = source.metadata_for("data/aws_docs/codepipeline/doc_source/concepts.md")

    assert metadata == {
        "url": "https://docs.aws.amazon.com/codepipeline/latest/userguide/concepts.html"
    }
    # rag_core refuses extra metadata that would clobber its own keys.
    assert not set(metadata) & RESERVED_METADATA_KEYS


def test_strip_anchors_removes_aws_heading_anchors():
    cleaned = _strip_anchors('# Build environments<a name="build-env"></a>\n\ntext')
    assert cleaned == "# Build environments\n\ntext"


def test_list_files_returns_markdown_and_strips_anchors_in_place(tmp_path, monkeypatch):
    """The clone/checkout half is stubbed; the file half is real."""
    source = AwsDocsGitSource(id="codebuild", path=str(tmp_path), **CODEBUILD_URLS)
    source.doc_dir.mkdir(parents=True)
    (source.doc_dir / "build-caching.md").write_text(
        '# Caching<a name="caching"></a>\n\ntext', encoding="utf-8"
    )
    (source.doc_dir / "concepts.md").write_text("# Concepts\n\ntext", encoding="utf-8")
    # Must be ignored: only *.md is ingested.
    (source.doc_dir / "notes.txt").write_text("ignore me", encoding="utf-8")

    monkeypatch.setattr(source, "_clone_and_checkout", lambda: None)
    files = source.list_files()

    assert [f.split("\\")[-1].split("/")[-1] for f in files] == [
        "build-caching.md",
        "concepts.md",
    ]
    # Anchors are stripped in the clone itself, so ingestion reads clean text.
    assert "<a name=" not in (source.doc_dir / "build-caching.md").read_text(encoding="utf-8")


def test_list_files_fails_loudly_when_nothing_was_recovered(tmp_path, monkeypatch):
    source = AwsDocsGitSource(id="codebuild", path=str(tmp_path), **CODEBUILD_URLS)
    source.doc_dir.mkdir(parents=True)
    monkeypatch.setattr(source, "_clone_and_checkout", lambda: None)

    with pytest.raises(RuntimeError, match="no \\*.md files"):
        source.list_files()

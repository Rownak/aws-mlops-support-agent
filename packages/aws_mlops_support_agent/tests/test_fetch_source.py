"""Tests for the awsdocs DocSource adapter.

The git-history recovery itself is verified by running the real ingestion (it
needs actual repos); what is tested here is the seam: the adapter satisfies
`DocSource`, is configured entirely from config.yml, and turns files into
LoadedDocs with correct provenance.
"""

import pytest
from aws_mlops_support_agent.sources.fetch import LOADERS, AwsDocsGitSource
from rag_core.config import SourceSpec
from rag_core.sources import DocSource, build_sources

SPEC = SourceSpec(
    id="codebuild",
    loader="awsdocs_git",
    options={
        "git_url": "https://github.com/awsdocs/aws-codebuild-user-guide.git",
        "docs_base_url": "https://docs.aws.amazon.com/codebuild/latest/userguide/",
    },
)


def test_adapter_satisfies_the_docsource_protocol():
    assert isinstance(AwsDocsGitSource(SPEC), DocSource)


def test_adapter_is_configured_from_the_spec():
    source = AwsDocsGitSource(SPEC)
    assert source.spec.id == "codebuild"
    assert source.git_url.endswith("aws-codebuild-user-guide.git")


def test_doc_url_maps_md_to_html():
    url = AwsDocsGitSource(SPEC).doc_url("build-env-ref-env-vars.md")
    assert (
        url == "https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref-env-vars.html"
    )


def test_missing_config_keys_fail_before_any_cloning():
    spec = SourceSpec(id="broken", loader="awsdocs_git", options={"git_url": "x"})
    with pytest.raises(RuntimeError, match="missing config.yml keys: docs_base_url"):
        AwsDocsGitSource(spec)


def test_fetch_yields_one_loaded_doc_per_markdown_file(tmp_path, monkeypatch):
    """clone_and_checkout is stubbed; fetch's own file-to-LoadedDoc half is real."""
    doc_dir = tmp_path / "doc_source"
    doc_dir.mkdir()
    (doc_dir / "build-caching.md").write_text("# Caching\n\ntext", encoding="utf-8")
    (doc_dir / "concepts.md").write_text("# Concepts\n\ntext", encoding="utf-8")

    source = AwsDocsGitSource(SPEC)
    monkeypatch.setattr(source, "clone_and_checkout", lambda: doc_dir)

    docs = list(source.fetch())
    assert [d.source_file for d in docs] == ["build-caching.md", "concepts.md"]
    assert all(d.source_id == "codebuild" for d in docs)
    assert docs[0].url.endswith("/build-caching.html")
    assert docs[0].text.startswith("# Caching")


def test_registry_wires_the_config_loader_name_to_this_adapter():
    (source,) = build_sources([SPEC], LOADERS)
    assert isinstance(source, AwsDocsGitSource)

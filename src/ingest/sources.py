"""The documentation corpus, declared as data.

This is the ONLY file to touch when adding a new documentation repo:
append a DocRepo entry to REPOS and re-run `uv run python -m src.ingest`.
The rest of the pipeline (fetch, chunk, index) is driven entirely by
these entries and never hardcodes a specific repo.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DocRepo:
    # Short slug used in chunk metadata, chunk IDs, and the local clone path.
    service: str
    # Repo to clone. awsdocs repos are archived with content stripped from
    # the default branch — fetch.py recovers the docs from git history.
    git_url: str
    # Prefix for reconstructing public doc URLs for citations:
    # doc_source/foo.md  ->  {docs_base_url}foo.html
    docs_base_url: str


REPOS: list[DocRepo] = [
    DocRepo(
        service="codebuild",
        git_url="https://github.com/awsdocs/aws-codebuild-user-guide.git",
        docs_base_url="https://docs.aws.amazon.com/codebuild/latest/userguide/",
    ),
    DocRepo(
        service="codepipeline",
        git_url="https://github.com/awsdocs/aws-codepipeline-user-guide.git",
        docs_base_url="https://docs.aws.amazon.com/codepipeline/latest/userguide/",
    ),
]

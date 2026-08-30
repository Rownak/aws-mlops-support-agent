"""Fetch awsdocs repos and recover the pre-archival doc_source content.

This is the AWS-specific half of ingestion, kept out of rag_core behind the
`DocSource` protocol (design.md §6). rag_core asks for documents; how they are
obtained — here, by digging them out of git history — is entirely our problem.

awsdocs repos are archived: the markdown was deleted from the default branch,
but it still lives in git history. Instead of matching commit-message text
(fragile), we use git itself to find where the docs last existed:

  1. If doc_source/ exists at HEAD, use HEAD as-is. This makes re-runs
     idempotent (a previous run left HEAD detached at the right commit) and
     supports future non-archived repos with zero changes.
  2. Otherwise, `git rev-list -1 HEAD -- doc_source` returns the most recent
     commit that TOUCHED doc_source/ — for an archived repo that is the
     deletion commit — so its parent (`<sha>^`) is the last commit where the
     docs were still present. Check that out (detached HEAD).
"""

import subprocess
from collections.abc import Iterator
from pathlib import Path

from rag_core.config import SourceSpec
from rag_core.sources import LoadedDoc

DATA_DIR = Path("data") / "repos"
DOC_DIR_NAME = "doc_source"

# The name this adapter registers under in config.yml's `loader:` field.
LOADER_NAME = "awsdocs_git"


def _git(repo_dir: Path, *args: str) -> str:
    """Run a git command inside repo_dir and return stripped stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _doc_dir_exists_at(repo_dir: Path, rev: str) -> bool:
    """True if doc_source/ exists in the tree of the given revision."""
    # ls-tree prints one line for the directory entry if it exists, else nothing.
    return bool(_git(repo_dir, "ls-tree", "-d", rev, DOC_DIR_NAME))


class AwsDocsGitSource:
    """A `rag_core.sources.DocSource` over one archived awsdocs repo.

    Configured entirely from a SourceSpec, so adding a third AWS guide means
    appending an entry to config.yml — no code change here.
    """

    def __init__(self, spec: SourceSpec, data_dir: Path = DATA_DIR):
        self.spec = spec
        self.data_dir = data_dir
        # Validated up front: a config typo should fail before any cloning.
        missing = [key for key in ("git_url", "docs_base_url") if not spec.options.get(key)]
        if missing:
            raise RuntimeError(
                f"Source '{spec.id}' (loader {LOADER_NAME}) is missing "
                f"config.yml keys: {', '.join(missing)}."
            )
        self.git_url = spec.options["git_url"]
        self.docs_base_url = spec.options["docs_base_url"]

    def doc_url(self, source_file: str) -> str:
        """doc_source/foo.md -> https://docs.aws.amazon.com/.../foo.html"""
        return f"{self.docs_base_url}{Path(source_file).stem}.html"

    def clone_and_checkout(self) -> Path:
        """Clone (if needed) and check out the docs. Returns the doc_source path."""
        repo_dir = self.data_dir / self.spec.id
        service = self.spec.id

        if repo_dir.exists():
            print(f"[fetch] {service}: clone exists at {repo_dir}, skipping clone")
        else:
            print(f"[fetch] {service}: cloning {self.git_url} (full history)")
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            # Full history (no --depth) — we need to reach the pre-archival commit.
            subprocess.run(
                ["git", "clone", "--quiet", self.git_url, str(repo_dir)],
                check=True,
            )

        if _doc_dir_exists_at(repo_dir, "HEAD"):
            print(f"[fetch] {service}: {DOC_DIR_NAME}/ present at HEAD")
        else:
            deletion_commit = _git(repo_dir, "rev-list", "-1", "HEAD", "--", DOC_DIR_NAME)
            if not deletion_commit:
                raise RuntimeError(
                    f"{service}: no commit in history ever touched {DOC_DIR_NAME}/ — "
                    "is docs_base layout different for this repo?"
                )
            pre_archival = f"{deletion_commit}^"
            if not _doc_dir_exists_at(repo_dir, pre_archival):
                raise RuntimeError(
                    f"{service}: {DOC_DIR_NAME}/ missing even at {pre_archival} — "
                    "unexpected history shape, inspect the repo manually."
                )
            sha = _git(repo_dir, "rev-parse", "--short", pre_archival)
            print(f"[fetch] {service}: checking out pre-archival commit {sha}")
            _git(repo_dir, "checkout", "--quiet", pre_archival)

        doc_dir = repo_dir / DOC_DIR_NAME
        md_count = len(list(doc_dir.glob("*.md")))
        if md_count == 0:
            raise RuntimeError(f"{service}: no *.md files in {doc_dir}")
        print(f"[fetch] {service}: {md_count} markdown files in {doc_dir}")
        return doc_dir

    def fetch(self) -> Iterator[LoadedDoc]:
        """The DocSource seam: yield every markdown file as a LoadedDoc."""
        doc_dir = self.clone_and_checkout()
        for md_file in sorted(doc_dir.glob("*.md")):
            yield LoadedDoc(
                source_id=self.spec.id,
                source_file=md_file.name,
                text=md_file.read_text(encoding="utf-8"),
                url=self.doc_url(md_file.name),
            )


# The project's loader registry: config.yml's `loader:` value -> adapter.
# `rag_core.sources.build_sources` consumes this; adding a corpus of a
# different shape means adding one entry here.
LOADERS = {LOADER_NAME: AwsDocsGitSource}

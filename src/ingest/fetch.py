"""Task 1.1 — Fetch doc repos and recover the pre-archival doc_source content.

awsdocs repos are archived: the markdown was deleted from the default branch,
but it still lives in git history. Instead of matching commit-message text
(fragile), we use git itself to find where the docs last existed:

  1. If doc_source/ exists at HEAD, use HEAD as-is. This makes re-runs
     idempotent (a previous run left HEAD detached at the right commit)
     and supports future non-archived repos with zero changes.
  2. Otherwise, `git rev-list -1 HEAD -- doc_source` returns the most
     recent commit that TOUCHED doc_source/ — for an archived repo that
     is the deletion commit — so its parent (`<sha>^`) is the last commit
     where the docs were still present. Check that out (detached HEAD).
"""

import subprocess
from pathlib import Path

from src.ingest.sources import DocRepo

DATA_DIR = Path("data") / "repos"
DOC_DIR_NAME = "doc_source"


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


def fetch_repo(repo: DocRepo, data_dir: Path = DATA_DIR) -> Path:
    """Clone (if needed) and check out the docs. Returns the doc_source path."""
    repo_dir = data_dir / repo.service

    if repo_dir.exists():
        print(f"[fetch] {repo.service}: clone exists at {repo_dir}, skipping clone")
    else:
        print(f"[fetch] {repo.service}: cloning {repo.git_url} (full history)")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        # Full history (no --depth) — we need to reach the pre-archival commit.
        subprocess.run(
            ["git", "clone", "--quiet", repo.git_url, str(repo_dir)],
            check=True,
        )

    if _doc_dir_exists_at(repo_dir, "HEAD"):
        print(f"[fetch] {repo.service}: {DOC_DIR_NAME}/ present at HEAD")
    else:
        deletion_commit = _git(repo_dir, "rev-list", "-1", "HEAD", "--", DOC_DIR_NAME)
        if not deletion_commit:
            raise RuntimeError(
                f"{repo.service}: no commit in history ever touched {DOC_DIR_NAME}/ — "
                "is docs_base layout different for this repo?"
            )
        pre_archival = f"{deletion_commit}^"
        if not _doc_dir_exists_at(repo_dir, pre_archival):
            raise RuntimeError(
                f"{repo.service}: {DOC_DIR_NAME}/ missing even at {pre_archival} — "
                "unexpected history shape, inspect the repo manually."
            )
        sha = _git(repo_dir, "rev-parse", "--short", pre_archival)
        print(f"[fetch] {repo.service}: checking out pre-archival commit {sha}")
        _git(repo_dir, "checkout", "--quiet", pre_archival)

    doc_dir = repo_dir / DOC_DIR_NAME
    md_count = len(list(doc_dir.glob("*.md")))
    if md_count == 0:
        raise RuntimeError(f"{repo.service}: no *.md files in {doc_dir}")
    print(f"[fetch] {repo.service}: {md_count} markdown files in {doc_dir}")
    return doc_dir

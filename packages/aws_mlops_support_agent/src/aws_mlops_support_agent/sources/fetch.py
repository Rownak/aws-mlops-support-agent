"""The awsdocs git source: a rag_core `Source` over an archived AWS user guide.

rag_core ships `local` and `s3` sources and knows nothing about git. This
module registers a third type, `awsdocs_git`, so `config.yml` can name it
directly and `RagCore.sync()` drives the whole thing — clone, recover the
pre-archival markdown, strip AWS's HTML anchor noise, list the files, and
hand each one its canonical docs URL via `metadata_for()`. That URL rides on
every chunk's `Document.metadata["url"]`, so citations read it straight off
a retrieved chunk with no sidecar file to keep in sync.

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

Anchor stripping rewrites the clone's own doc_source/*.md in place, so the
working tree of the clone reads as dirty. That is deliberate: the clone is a
cache, not something we commit from, and it avoids keeping a second copy of
every file.
"""

import logging
import re
import subprocess
from pathlib import Path

from rag_core.sources import REGISTRY, Source

logger = logging.getLogger(__name__)

DOC_DIR_NAME = "doc_source"

#: AWS anchors every heading with a raw HTML anchor, e.g.
#: `# Build environments<a name="build-env"></a>`. Stripping them keeps
#: citations clean and removes noise from the embeddings.
STRIP_PATTERNS = [r'<a name="[^"]*"></a>']


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


def _strip_anchors(text: str) -> str:
    for pattern in STRIP_PATTERNS:
        text = re.sub(pattern, "", text)
    return text


class AwsDocsGitSource(Source):
    """One archived awsdocs repo as a rag_core source.

    Config.yml entry:

    .. code-block:: yaml

        sources:
          - type: awsdocs_git
            id: codebuild
            path: data/aws_docs      # where clones live
            git_url: https://github.com/awsdocs/aws-codebuild-user-guide.git
            docs_base_url: https://docs.aws.amazon.com/codebuild/latest/userguide/

    Attributes:
        id: Names this guide (used for the clone's directory name and in logs)
        repo_dir: The local clone
        doc_dir: The recovered markdown inside that clone
    """

    type_name = "awsdocs_git"

    def __init__(
        self,
        id: str,
        git_url: str,
        docs_base_url: str,
        path: str = "data/aws_docs",
        extensions: list[str] | None = None,
        name: str = "",
        **_ignored,
    ):
        super().__init__(name=name or f"awsdocs:{id}", extensions=extensions or [".md"])
        self.id = id
        self.git_url = git_url
        self.docs_base_url = docs_base_url
        self.repo_dir = Path(path) / id
        self.doc_dir = self.repo_dir / DOC_DIR_NAME

    def doc_url(self, file_path: str) -> str:
        """doc_source/foo.md -> https://docs.aws.amazon.com/.../foo.html"""
        return f"{self.docs_base_url}{Path(file_path).stem}.html"

    def metadata_for(self, file_path: str) -> dict:
        """The canonical AWS docs URL for this file, carried on every chunk.

        This is what the sidecar manifest used to do — except the value now
        travels with the chunk, so nothing has to re-derive it from a path
        after retrieval.
        """
        return {"url": self.doc_url(file_path)}

    def _clone_and_checkout(self) -> None:
        """Clone if needed and check out the revision where the docs exist."""
        if self.repo_dir.exists():
            logger.info(f"{self.name}: clone exists at {self.repo_dir}, skipping clone")
        else:
            logger.info(f"{self.name}: cloning {self.git_url} (full history)")
            self.repo_dir.parent.mkdir(parents=True, exist_ok=True)
            # Full history (no --depth) — we need to reach the pre-archival commit.
            subprocess.run(
                ["git", "clone", "--quiet", self.git_url, str(self.repo_dir)],
                check=True,
            )

        if _doc_dir_exists_at(self.repo_dir, "HEAD"):
            logger.info(f"{self.name}: {DOC_DIR_NAME}/ present at HEAD")
            return

        deletion_commit = _git(self.repo_dir, "rev-list", "-1", "HEAD", "--", DOC_DIR_NAME)
        if not deletion_commit:
            raise RuntimeError(
                f"{self.name}: no commit in history ever touched {DOC_DIR_NAME}/ — "
                "is docs_base layout different for this repo?"
            )
        pre_archival = f"{deletion_commit}^"
        if not _doc_dir_exists_at(self.repo_dir, pre_archival):
            raise RuntimeError(
                f"{self.name}: {DOC_DIR_NAME}/ missing even at {pre_archival} — "
                "unexpected history shape, inspect the repo manually."
            )
        sha = _git(self.repo_dir, "rev-parse", "--short", pre_archival)
        logger.info(f"{self.name}: checking out pre-archival commit {sha}")
        _git(self.repo_dir, "checkout", "--quiet", pre_archival)

    def list_files(self) -> list[str]:
        """
        Recover this guide's markdown and return the files to ingest.

        Clones (or reuses) the repo, checks out the revision where the docs
        still existed, and strips AWS's HTML anchors in place.

        Returns:
            Paths of the recovered markdown, sorted so a run is reproducible

        Raises:
            RuntimeError: If the repo's history has no recoverable docs
        """
        self._clone_and_checkout()

        md_files = sorted(self.doc_dir.glob("*.md"))
        if not md_files:
            raise RuntimeError(f"{self.name}: no *.md files in {self.doc_dir}")

        stripped = 0
        for md_file in md_files:
            original = md_file.read_text(encoding="utf-8")
            cleaned = _strip_anchors(original)
            # Only rewrite what actually changed, so a re-run leaves the
            # mtimes (and the clone) alone once everything is already clean.
            if cleaned != original:
                md_file.write_text(cleaned, encoding="utf-8")
                stripped += 1

        logger.info(
            f"{self.name}: {len(md_files)} markdown file(s) in {self.doc_dir} "
            f"({stripped} rewritten by anchor stripping)"
        )
        return [str(p) for p in md_files]


# rag_core dispatches config.yml's `type:` through this registry, so importing
# this module is what makes `type: awsdocs_git` resolvable.
REGISTRY.register(AwsDocsGitSource)

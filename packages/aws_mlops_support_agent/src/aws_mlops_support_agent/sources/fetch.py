"""Fetch awsdocs repos and recover the pre-archival doc_source content.

This is the AWS-specific half of ingestion. rag_core only understands
`type: local` sources (a folder of files); it knows nothing about git or
awsdocs. This module is what makes those folders exist: it clones each
configured repo, recovers the pre-archival markdown from git history, strips
AWS's HTML anchor noise, and writes the result to the path each config.yml
`sources` entry points at — plus a `_manifest.json` sidecar mapping each
filename to its canonical AWS docs URL, since rag_core's chunker has no slot
for a custom `url` field. `doc_url_for()` below is the read side, used by
`agent.ticket` to render citations from a chunk's `metadata["source"]`.

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

Run directly:  uv run python -m aws_mlops_support_agent.sources.fetch
"""

import json
import re
import subprocess
from pathlib import Path

from rag_core.config import SourceSpec

DOC_DIR_NAME = "doc_source"
MANIFEST_NAME = "_manifest.json"

#: AWS anchors every heading with a raw HTML anchor, e.g.
#: `# Build environments<a name="build-env"></a>`. Stripping them keeps
#: citations clean and removes noise from the embeddings.
STRIP_PATTERNS = [r'<a name="[^"]*"></a>']

# Per-repo git/doc info. Not in config.yml: rag_core's `sources` schema has no
# room for it (a `local` source is just a path), and these are AWS-specific
# facts about how to populate that path, not something the generic engine
# should ever need to know.
AWSDOCS_REPOS = {
    "codebuild": {
        "git_url": "https://github.com/awsdocs/aws-codebuild-user-guide.git",
        "docs_base_url": "https://docs.aws.amazon.com/codebuild/latest/userguide/",
    },
    "codepipeline": {
        "git_url": "https://github.com/awsdocs/aws-codepipeline-user-guide.git",
        "docs_base_url": "https://docs.aws.amazon.com/codepipeline/latest/userguide/",
    },
}


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


class AwsDocsGitSource:
    """Recovers one archived awsdocs repo's markdown to a local directory.

    Configured from a `sources` config.yml entry (a `type: local` block plus
    an `id` rag_core ignores) and `AWSDOCS_REPOS[id]`. Adding a third AWS
    guide means one entry in each place — no other code changes.
    """

    def __init__(self, spec: SourceSpec, clone_root: Path):
        self.spec = spec
        self.id = spec.options["id"]
        if self.id not in AWSDOCS_REPOS:
            raise RuntimeError(
                f"Source '{self.id}' has no entry in AWSDOCS_REPOS "
                f"(sources/fetch.py). Available: {', '.join(sorted(AWSDOCS_REPOS))}."
            )
        self.git_url = AWSDOCS_REPOS[self.id]["git_url"]
        self.docs_base_url = AWSDOCS_REPOS[self.id]["docs_base_url"]
        # e.g. data/aws_docs/codebuild — the clone lives one level above the
        # doc_source/ path that config.yml's `local` source points at.
        self.repo_dir = clone_root / self.id
        # config.yml's source `path` (data/aws_docs/codebuild/doc_source) is
        # the destination the recovered files must land in.
        self.doc_dir = Path(spec.options["path"])

    def doc_url(self, source_file: str) -> str:
        """doc_source/foo.md -> https://docs.aws.amazon.com/.../foo.html"""
        return f"{self.docs_base_url}{Path(source_file).stem}.html"

    def _clone_and_checkout(self) -> Path:
        """Clone (if needed) and check out the docs. Returns the recovered doc_source path."""
        if self.repo_dir.exists():
            print(f"[fetch] {self.id}: clone exists at {self.repo_dir}, skipping clone")
        else:
            print(f"[fetch] {self.id}: cloning {self.git_url} (full history)")
            self.repo_dir.parent.mkdir(parents=True, exist_ok=True)
            # Full history (no --depth) — we need to reach the pre-archival commit.
            subprocess.run(
                ["git", "clone", "--quiet", self.git_url, str(self.repo_dir)],
                check=True,
            )

        if _doc_dir_exists_at(self.repo_dir, "HEAD"):
            print(f"[fetch] {self.id}: {DOC_DIR_NAME}/ present at HEAD")
        else:
            deletion_commit = _git(self.repo_dir, "rev-list", "-1", "HEAD", "--", DOC_DIR_NAME)
            if not deletion_commit:
                raise RuntimeError(
                    f"{self.id}: no commit in history ever touched {DOC_DIR_NAME}/ — "
                    "is docs_base layout different for this repo?"
                )
            pre_archival = f"{deletion_commit}^"
            if not _doc_dir_exists_at(self.repo_dir, pre_archival):
                raise RuntimeError(
                    f"{self.id}: {DOC_DIR_NAME}/ missing even at {pre_archival} — "
                    "unexpected history shape, inspect the repo manually."
                )
            sha = _git(self.repo_dir, "rev-parse", "--short", pre_archival)
            print(f"[fetch] {self.id}: checking out pre-archival commit {sha}")
            _git(self.repo_dir, "checkout", "--quiet", pre_archival)

        recovered_dir = self.repo_dir / DOC_DIR_NAME
        md_count = len(list(recovered_dir.glob("*.md")))
        if md_count == 0:
            raise RuntimeError(f"{self.id}: no *.md files in {recovered_dir}")
        print(f"[fetch] {self.id}: {md_count} markdown files in {recovered_dir}")
        return recovered_dir

    def fetch(self) -> None:
        """Recover the docs, strip anchors, and write .md files + manifest to doc_dir."""
        recovered_dir = self._clone_and_checkout()
        self.doc_dir.mkdir(parents=True, exist_ok=True)

        manifest: dict[str, str] = {}
        for md_file in sorted(recovered_dir.glob("*.md")):
            cleaned = _strip_anchors(md_file.read_text(encoding="utf-8"))
            (self.doc_dir / md_file.name).write_text(cleaned, encoding="utf-8")
            manifest[md_file.name] = self.doc_url(md_file.name)

        manifest_path = self.doc_dir / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[fetch] {self.id}: wrote {len(manifest)} file(s) to {self.doc_dir}")


#: Manifests are small and re-read often (once per cited chunk); caching by
#: directory avoids re-parsing the same JSON file for every citation.
_manifest_cache: dict[Path, dict[str, str]] = {}


def _load_manifest(doc_dir: Path) -> dict[str, str]:
    if doc_dir not in _manifest_cache:
        manifest_path = doc_dir / MANIFEST_NAME
        _manifest_cache[doc_dir] = (
            json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        )
    return _manifest_cache[doc_dir]


def doc_url_for(source: str) -> str | None:
    """Look up the AWS docs URL for a chunk's `metadata["source"]` path.

    Reads the `_manifest.json` `fetch()` wrote alongside the file. Returns
    None when there is no manifest (e.g. a non-awsdocs local file, or a test
    fixture) rather than raising — a missing citation URL is not fatal.
    """
    path = Path(source)
    return _load_manifest(path.parent).get(path.name)


def data_dir_from_sources(sources: tuple[SourceSpec, ...]) -> Path:
    """The directory every configured `local` source's clone lives under.

    Derived from config.yml rather than hardcoded, so the layout is defined
    in exactly one place: the first two path components of a source's `path`
    (e.g. "data/aws_docs/codebuild/doc_source" -> "data/aws_docs").
    """
    if not sources:
        raise RuntimeError("No sources configured in config.yml.")
    first_path = Path(sources[0].options["path"]).parts
    if len(first_path) < 2:
        raise RuntimeError(
            f"Source path '{sources[0].options['path']}' has fewer than 2 "
            "components; expected e.g. 'data/aws_docs/<id>/doc_source'."
        )
    return Path(*first_path[:2])


def fetch_all(sources: tuple[SourceSpec, ...]) -> None:
    """Recover every configured awsdocs repo's markdown to its source path.

    Safe to re-run: clones and checkouts are reused, and files are simply
    overwritten with freshly stripped content each time.
    """
    clone_root = data_dir_from_sources(sources)
    for spec in sources:
        AwsDocsGitSource(spec, clone_root).fetch()


if __name__ == "__main__":
    from aws_mlops_support_agent.settings import load_settings

    fetch_all(load_settings().rag.sources)

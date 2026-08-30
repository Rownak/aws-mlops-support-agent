"""Enforce the one-way dependency rule (design.md §6).

`rag_core` is the reusable engine; a project package (aws_mlops_support_agent,
later scifact_rag) may import it, never the reverse. A single accidental
`from aws_mlops_support_agent...` inside rag_core would silently destroy the
reusability the whole workspace split exists for — and nothing else would fail,
because both packages are installed together in development.

So the rule is checked structurally: parse every rag_core source file and look
at what it imports. AST rather than grep, so a project name inside a string or
comment (like this docstring) can't trigger a false positive.
"""

import ast
from pathlib import Path

import pytest

# Distribution names that are PROJECTS, not the engine. Any import rooted at
# one of these is a boundary violation. A new project package gets added here.
PROJECT_PACKAGES = {"aws_mlops_support_agent", "scifact_rag"}

RAG_CORE_PKG = Path(__file__).resolve().parents[1]  # packages/rag_core/
RAG_CORE_SRC = RAG_CORE_PKG / "src" / "rag_core"


def imported_roots(tree: ast.AST) -> set[str]:
    """Every top-level package name imported by a module.

    Handles both `import a.b` and `from a.b import c`. Relative imports
    (`from .x import y`) have no module root to check, so they're skipped —
    they can only ever point inside rag_core anyway.
    """
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # node.level > 0 means a relative import; node.module is then
            # relative to this package and cannot reach a project package.
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def source_files() -> list[Path]:
    return sorted(RAG_CORE_SRC.rglob("*.py"))


def test_rag_core_sources_were_found():
    """Guard the guard: an empty file list would make every check below vacuous."""
    files = source_files()
    assert files, f"no rag_core sources found under {RAG_CORE_SRC}"
    assert len(files) >= 10


@pytest.mark.parametrize("path", source_files(), ids=lambda p: p.name)
def test_module_does_not_import_a_project_package(path: Path):
    """The rule itself, reported per-file so a failure names the culprit."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations = imported_roots(tree) & PROJECT_PACKAGES
    assert not violations, (
        f"{path.relative_to(RAG_CORE_PKG)} imports {', '.join(sorted(violations))}. "
        "rag_core must never depend on a project package (design.md §6) — "
        "invert the dependency: have the project pass what it needs in."
    )


def test_declared_dependencies_contain_no_project_package():
    """The same rule at the packaging layer, not just the import layer."""
    pyproject = (RAG_CORE_PKG / "pyproject.toml").read_text(encoding="utf-8")
    for name in PROJECT_PACKAGES:
        # Distribution names use hyphens; the import name uses underscores.
        assert name.replace("_", "-") not in pyproject, (
            f"rag-core declares a dependency on {name} — the engine must be installable on its own."
        )


# --- the detector itself must actually detect ---


def test_detector_catches_a_plain_import():
    tree = ast.parse("import aws_mlops_support_agent.agent.graph\n")
    assert imported_roots(tree) & PROJECT_PACKAGES


def test_detector_catches_a_from_import():
    tree = ast.parse("from aws_mlops_support_agent.settings import AgentConfig\n")
    assert imported_roots(tree) & PROJECT_PACKAGES


def test_detector_ignores_project_names_in_strings_and_comments():
    """Why AST and not grep: prose mentioning a project must not fail the build."""
    tree = ast.parse('# see aws_mlops_support_agent\nX = "aws_mlops_support_agent"\n')
    assert not imported_roots(tree) & PROJECT_PACKAGES


def test_detector_allows_normal_engine_imports():
    tree = ast.parse("from rag_core.config import RagConfig\nimport yaml\n")
    assert not imported_roots(tree) & PROJECT_PACKAGES

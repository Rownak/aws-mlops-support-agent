"""Tests for the DocSource seam.

The point of these tests: prove a corpus adapter needs nothing from rag_core
beyond the protocol — an in-memory fake with no I/O at all satisfies it.
"""

import pytest
from rag_core.config import SourceSpec
from rag_core.sources import DocSource, LoadedDoc, build_sources


class FakeSource:
    """A DocSource that serves documents from a dict. No network, no disk."""

    def __init__(self, spec: SourceSpec, docs: dict[str, str] | None = None):
        self.spec = spec
        self._docs = docs or {"intro.md": "# Intro\n\nhello"}

    def fetch(self):
        for name, text in self._docs.items():
            yield LoadedDoc(
                source_id=self.spec.id,
                source_file=name,
                text=text,
                url=f"https://docs.example.invalid/{name.removesuffix('.md')}.html",
            )


SPEC = SourceSpec(id="fake", loader="fake_loader", options={"anything": "goes"})


def test_fake_source_satisfies_the_protocol():
    assert isinstance(FakeSource(SPEC), DocSource)


def test_object_without_fetch_does_not_satisfy_the_protocol():
    class NotASource:
        spec = SPEC

    assert not isinstance(NotASource(), DocSource)


def test_fetch_yields_loaded_docs_with_provenance():
    (doc,) = list(FakeSource(SPEC).fetch())
    assert doc.source_id == "fake"
    assert doc.source_file == "intro.md"
    assert doc.text.startswith("# Intro")
    assert doc.url.endswith("/intro.html")
    assert doc.extra == {}


def test_build_sources_uses_the_projects_registry():
    specs = [SPEC, SourceSpec(id="other", loader="fake_loader")]
    registry = {"fake_loader": FakeSource}
    sources = list(build_sources(specs, registry))
    assert [s.spec.id for s in sources] == ["fake", "other"]
    assert all(isinstance(s, DocSource) for s in sources)


def test_unknown_loader_fails_with_the_known_names():
    spec = SourceSpec(id="x", loader="typo_loader")
    with pytest.raises(RuntimeError, match="typo_loader.*Known loaders: fake_loader"):
        list(build_sources([spec], {"fake_loader": FakeSource}))


def test_unknown_loader_message_handles_an_empty_registry():
    spec = SourceSpec(id="x", loader="anything")
    with pytest.raises(RuntimeError, match="none registered"):
        list(build_sources([spec], {}))

"""Ingestion entrypoint:  uv run aws-agent-ingest

Wires this project's pieces into rag_core's pipeline: the sources come from
config.yml, the adapters from our loader registry, and everything after that
— chunk, embed, upsert — is the engine's generic path.

Safe to re-run: clones are reused and upserts use deterministic IDs.
"""

from rag_core.observability import setup_json_logging
from rag_core.pipeline import run_ingest
from rag_core.sources import build_sources

from aws_mlops_support_agent.settings import load_settings
from aws_mlops_support_agent.sources.fetch import LOADERS


def main() -> None:
    setup_json_logging()
    cfg = load_settings()

    # build_sources maps each config.yml `loader:` name onto an adapter; the
    # engine never learns what an awsdocs repo is.
    sources = build_sources(cfg.rag.sources, LOADERS)

    report = run_ingest(cfg.rag, sources)
    print(report.summary())


if __name__ == "__main__":
    main()

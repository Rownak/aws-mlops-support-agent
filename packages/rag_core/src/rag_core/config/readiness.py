"""Live reachability checks for local-dev backends (Ollama, Pinecone Local).

Separate from `providers.py`, which is pure config parsing with no I/O — this
module is the one place that makes network calls, kept small and easy to skip
(`RAG_SKIP_READINESS_CHECK`) for CI/offline use. Cloud providers (OpenAI,
Google, managed Pinecone) are not checked here; their reachability is assumed
and only their credentials are validated, by `missing_secrets()`.
"""

import os

import httpx

_READINESS_TIMEOUT_S = 2.0


def _skip_readiness_check() -> bool:
    return os.environ.get("RAG_SKIP_READINESS_CHECK", "").lower() in ("1", "true", "yes")


def check_ollama_ready(base_url: str, model: str) -> list[str]:
    """Confirm Ollama is reachable and `model` has been pulled."""
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=_READINESS_TIMEOUT_S)
        response.raise_for_status()
    except httpx.HTTPError:
        return [f"Ollama not reachable at {base_url}"]

    names = {m.get("model") or m.get("name") for m in response.json().get("models", [])}
    if model not in names:
        return [f"Ollama model '{model}' not found — run: ollama pull {model}"]
    return []


def check_pinecone_local_ready(host: str) -> list[str]:
    """Confirm something is listening at `host` (Pinecone Local's control port).

    Pinecone Local has no root route, so even a 404 proves it's up — only a
    connection failure counts as not-ready.
    """
    try:
        httpx.get(host, timeout=_READINESS_TIMEOUT_S)
    except httpx.HTTPError:
        return [
            f"Pinecone Local not reachable at {host} — run: "
            "docker run -d --name pinecone-local -e PORT=5080 -e PINECONE_HOST=localhost "
            "-p 5080-5090:5080-5090 ghcr.io/pinecone-io/pinecone-local:latest"
        ]
    return []


def check_readiness(config) -> None:
    """Raise one RuntimeError listing every unready backend, or return.

    Sibling to `RagConfig._validate()`'s `missing_secrets()` aggregation, kept
    separate because this does live I/O and that one must stay pure — see
    `RagCore.__init__`, which calls this after `load_config` instead of
    `load_config` calling it itself.
    """
    if _skip_readiness_check():
        return

    problems: list[str] = []
    for block in (config.embeddings, config.llm, config.vectorstore):
        check = getattr(block, "missing_readiness", None)
        if check:
            problems.extend(check())

    if problems:
        raise RuntimeError("rag_core is not ready:\n- " + "\n- ".join(problems))

"""Environment-variable plumbing shared by every config dataclass.

The lowest layer of the config package: stdlib only, imports nothing from
rag_core, so every other config module can depend on it freely.
"""

import os

# provider -> env var its API key is read from. Ollama and a local/hosted
# Pinecone instance need no key, so they are absent here rather than mapped
# to something that would always be "missing".
_EMBEDDING_PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}
_LLM_PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


def _env_or(name: str, yaml_value, default):
    """Apply the precedence rule: env var > config.yaml > built-in default.

    `os.environ.get(name) or ...` (not `.get(name, ...)`) so that an empty
    string in .env still falls through to the next layer.

    Note `name` may be "" — callers pass the result of a provider-map lookup
    for a provider that needs no key (ollama). `os.environ.get("")` is None,
    so it falls through correctly; do not "clean up" to `os.environ[name]`.
    """
    return os.environ.get(name) or yaml_value or default

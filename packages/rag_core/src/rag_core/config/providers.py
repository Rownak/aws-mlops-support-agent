"""Config blocks for the three pluggable providers: embeddings, LLM, vector store.

Grouped together because they share an idiom the other blocks do not: each
selects a `provider`, each may need an API key, and each renders itself as a
plain dict for its factory. Adding a provider should mean editing this file
and nothing else.
"""

from dataclasses import dataclass

from .base import DictLike
from .env import _EMBEDDING_PROVIDER_KEY_ENV, _LLM_PROVIDER_KEY_ENV, _env_or
from .readiness import check_ollama_ready, check_pinecone_local_ready


@dataclass(frozen=True)
class EmbeddingConfig(DictLike):
    provider: str = "ollama"
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    api_key: str | None = None

    #: Masked by `describe()`. Read via getattr, so only classes that set it pay.
    _secret_fields = frozenset({"api_key"})

    @classmethod
    def from_raw(cls, raw: dict) -> "EmbeddingConfig":
        """Parse the ``embeddings`` block. The API key comes from env only."""
        provider = raw.get("provider") or cls.provider
        return cls(
            provider=provider,
            model=raw.get("model") or cls.model,
            base_url=raw.get("base_url") or cls.base_url,
            # "" for a provider that needs no key; _env_or handles that.
            api_key=_env_or(
                _EMBEDDING_PROVIDER_KEY_ENV.get(provider, ""), raw.get("api_key"), None
            ),
        )

    def missing_secrets(self) -> list[str]:
        """Which required env vars are absent, as human-readable strings."""
        env = _EMBEDDING_PROVIDER_KEY_ENV.get(self.provider)
        if env and not self.api_key:
            return [f"{env} (required by embeddings.provider: {self.provider})"]
        return []

    def missing_readiness(self) -> list[str]:
        """Only ollama has a local server/model to verify reachability for."""
        if self.provider == "ollama":
            return check_ollama_ready(self.base_url, self.model)
        return []


@dataclass(frozen=True)
class LLMConfig(DictLike):
    provider: str = "ollama"
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    api_key: str | None = None
    num_ctx: int | None = None

    _secret_fields = frozenset({"api_key"})
    # Dropped when None so ChatOllama/ChatOpenAI apply their own default.
    _optional = frozenset({"num_ctx"})

    @classmethod
    def from_raw(cls, raw: dict) -> "LLMConfig":
        """Parse the ``llm`` block. The API key comes from env only."""
        provider = raw.get("provider") or cls.provider
        return cls(
            provider=provider,
            model=raw.get("model") or cls.model,
            base_url=raw.get("base_url") or cls.base_url,
            api_key=_env_or(_LLM_PROVIDER_KEY_ENV.get(provider, ""), raw.get("api_key"), None),
            num_ctx=raw.get("num_ctx"),
        )

    def missing_secrets(self) -> list[str]:
        env = _LLM_PROVIDER_KEY_ENV.get(self.provider)
        if env and not self.api_key:
            return [f"{env} (required by llm.provider: {self.provider})"]
        return []

    def missing_readiness(self) -> list[str]:
        """Only ollama has a local server/model to verify reachability for."""
        if self.provider == "ollama":
            return check_ollama_ready(self.base_url, self.model)
        return []


@dataclass(frozen=True)
class VectorStoreConfig(DictLike):
    provider: str = "pinecone"
    collection_name: str = "rag-docs"
    use_sparse: bool = False
    api_key: str | None = None
    host: str | None = None
    cloud: str = "aws"
    region: str = "us-east-1"

    _secret_fields = frozenset({"api_key"})
    # Dropped when None: its presence is what selects Pinecone Local.
    _optional = frozenset({"host"})

    @classmethod
    def from_raw(cls, raw: dict) -> "VectorStoreConfig":
        """Parse the ``vectorstore`` block.

        Unlike the other two this has a single backend, so the key env var is
        named directly rather than looked up in a provider map.
        """
        return cls(
            provider=raw.get("provider") or cls.provider,
            collection_name=raw.get("collection_name") or cls.collection_name,
            # Not via _env_or: `False` is falsy, so `or` would flip it to the
            # default. Read straight from the raw block instead.
            use_sparse=bool(raw.get("use_sparse", False)),
            api_key=_env_or("PINECONE_API_KEY", raw.get("api_key"), None),
            host=raw.get("host"),
            cloud=raw.get("cloud") or cls.cloud,
            region=_env_or("AWS_REGION", raw.get("region"), cls.region),
        )

    def missing_secrets(self) -> list[str]:
        """Pinecone needs a key only when talking to the managed service.

        `host` set means Pinecone Local or a self-hosted instance, which
        needs no key — demanding one would make local dev impossible.
        """
        if self.provider == "pinecone" and not self.host and not self.api_key:
            return [
                "PINECONE_API_KEY (required by vectorstore.provider: pinecone, unless "
                "vectorstore.host is set to use a local/self-hosted instance)"
            ]
        return []

    def missing_readiness(self) -> list[str]:
        """Only Pinecone Local (host set) has something to verify is up.

        Managed Pinecone Cloud is assumed reachable — its credentials are
        checked by `missing_secrets()` instead.
        """
        if self.provider == "pinecone" and self.host:
            return check_pinecone_local_ready(self.host)
        return []

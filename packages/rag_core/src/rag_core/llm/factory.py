"""
Chat-model factory for multiple provider support.

Mirrors ``embeddings/factory.py``: one function reads a `provider` key from
config and returns a LangChain chat model, so `generation.generator` never
imports a specific vendor's SDK directly.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_llm(config: dict, **kwargs: Any) -> Any:
    """
    Create a chat model instance from configuration.

    Args:
        config: Configuration dictionary with 'provider' key and
               provider-specific settings
        **kwargs: Additional keyword arguments to pass to the constructor

    Returns:
        Initialized chat model instance

    Raises:
        ValueError: If provider is not supported
        ImportError: If the provider's package is not installed

    Example:
        >>> llm = get_llm({"provider": "ollama", "model": "llama3.1:8b"})  # doctest: +SKIP
        >>> llm = get_llm({"provider": "openai", "model": "gpt-4o-mini"})  # doctest: +SKIP
    """
    provider = config.get("provider", "").lower()

    try:
        if provider == "openai":
            return _get_openai_llm(config, **kwargs)
        elif provider == "ollama":
            return _get_ollama_llm(config, **kwargs)
        elif provider == "google" or provider == "gemini":
            return _get_google_llm(config, **kwargs)
        else:
            valid = "ollama, openai, google"
            raise ValueError(
                f"Unsupported LLM provider: '{provider}'. Valid options: {valid}\n"
                f"Example config:\n"
                f"  llm:\n"
                f"    provider: ollama\n"
                f"    model: llama3.1:8b"
            )
    except ImportError as e:
        logger.error(f"Missing dependency for {provider} LLM: {e}")
        raise ImportError(
            f"Required package for '{provider}' LLM is not installed.\n"
            f"Run: {get_install_command(provider)}"
        )


def _get_openai_llm(config: dict, **kwargs) -> Any:
    from langchain_openai import ChatOpenAI

    model = config.get("model", "gpt-4o-mini")
    return ChatOpenAI(
        model=model,
        temperature=config.get("temperature", 0),
        api_key=config.get("api_key"),
        **kwargs,
    )


def _get_ollama_llm(config: dict, **kwargs) -> Any:
    from langchain_ollama import ChatOllama

    model = config.get("model", "llama3.1:8b")
    base_url = config.get("base_url", "http://localhost:11434")
    extra = {}
    if "num_ctx" in config:
        extra["num_ctx"] = config["num_ctx"]

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=config.get("temperature", 0),
        **extra,
        **kwargs,
    )


def _get_google_llm(config: dict, **kwargs) -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = config.get("model", "gemini-1.5-flash")
    return ChatGoogleGenerativeAI(
        model=model, google_api_key=config.get("api_key"), **kwargs
    )


def get_install_command(provider: str) -> str:
    """Get the pip install command for a provider."""
    commands = {
        "openai": "pip install langchain-openai",
        "ollama": "pip install langchain-ollama",
        "google": "pip install langchain-google-genai",
    }
    return commands.get(provider, "pip install langchain-community")

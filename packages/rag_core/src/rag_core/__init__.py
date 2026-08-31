"""Corpus-agnostic retrieval-augmented-generation engine."""

from .config import Config
from .pipeline import RagCore

__all__ = ["Config", "RagCore"]

"""Disk embedding cache: keyed on model + corpus hash, doc_id order preserved."""

import numpy as np
from rag_bench_eval import embedding_cache


def test_miss_when_nothing_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_cache, "EMBEDDINGS_CACHE_DIR", tmp_path)
    assert embedding_cache.load("some-model", {"d1": "text"}) is None


def test_save_then_load_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_cache, "EMBEDDINGS_CACHE_DIR", tmp_path)
    corpus = {"d1": "cats are great", "d2": "dogs are great"}
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])

    embedding_cache.save("model-a", corpus, list(corpus.keys()), vectors)
    cached = embedding_cache.load("model-a", corpus)

    assert cached is not None
    doc_ids, loaded_vectors = cached
    assert doc_ids == ["d1", "d2"]
    np.testing.assert_array_equal(loaded_vectors, vectors)


def test_corpus_change_invalidates_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_cache, "EMBEDDINGS_CACHE_DIR", tmp_path)
    corpus = {"d1": "cats are great"}
    vectors = np.array([[1.0, 0.0]])
    embedding_cache.save("model-a", corpus, list(corpus.keys()), vectors)

    changed_corpus = {"d1": "cats are wonderful"}
    assert embedding_cache.load("model-a", changed_corpus) is None


def test_different_model_is_a_separate_cache_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_cache, "EMBEDDINGS_CACHE_DIR", tmp_path)
    corpus = {"d1": "cats are great"}
    vectors = np.array([[1.0, 0.0]])
    embedding_cache.save("model-a", corpus, list(corpus.keys()), vectors)

    assert embedding_cache.load("model-b", corpus) is None

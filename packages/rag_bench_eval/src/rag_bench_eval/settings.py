"""Paths shared across rag_bench_eval: dataset cache and results output."""

from pathlib import Path

# Package root (packages/rag_bench_eval/), so paths resolve the same
# regardless of the caller's working directory.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PACKAGE_ROOT / "data"
NFCORPUS_DIR = DATA_DIR / "beir" / "nfcorpus"
CQADUPSTACK_PROGRAMMERS_DIR = DATA_DIR / "cqadupstack" / "programmers"
RAGBENCH_EMANUAL_DIR = DATA_DIR / "ragbench" / "emanual"

# Shared by every dataset's disk-persisted embedding vectors (embedding_cache.py).
# Filenames are keyed by model + corpus hash, so datasets never collide here.
EMBEDDINGS_CACHE_DIR = DATA_DIR / ".embeddings"

RESULTS_DIR = PACKAGE_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"

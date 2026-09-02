"""Load benchmark.yaml into plain dicts (design_summary.md §Config).

No schema layer: an unknown pipeline `type` raises where it's dispatched
(build_retriever, phase 2.7+), and a missing resource name raises at lookup
(resources.py). This module only reads the file.
"""

from pathlib import Path

import yaml

from rag_bench_eval.settings import PACKAGE_ROOT

CONFIG_PATH = PACKAGE_ROOT / "benchmark.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

# tasks v3.0 — rag_bench_eval, phases 0–1

Retrieval-optimization benchmark on BEIR/NFCorpus. Design: `design.md`.
One task at a time; each phase ends runnable.

Phases 2–8 are in `design.md` §6 and get written up once phase 1 clears its gate.

---

## Phase 0 — Scaffold

- [ ] **0.1 Package skeleton.** `packages/rag_bench_eval/` with
  `pyproject.toml` (name `rag-bench-eval`, depends on `rag-core`),
  `src/rag_bench_eval/__init__.py`, `tests/`.
  *Done when:* `uv sync` succeeds and `import rag_bench_eval` works.

- [ ] **0.2 Workspace registration.** Add to the root `pyproject.toml`:
  `[tool.uv.sources]` entry and `[tool.pytest.ini_options] testpaths`.
  *Done when:* `uv run pytest` from the root collects this package's tests.

- [ ] **0.3 Settings.** `settings.py` with the dataset cache dir
  (`data/beir/nfcorpus/`) and results dir. Add `data/beir/` to `.gitignore` —
  NFCorpus is CC BY-SA, fetched at run time, never committed.

---

## Phase 1 — End-to-end: corpus → BM25 → nDCG@10

The gate. BM25 near the published nDCG@10 ≈ 0.32 validates dataset parsing,
qrels handling and the metric against one external number. Nothing later is
trustworthy until it does, so build the thinnest path that produces it.

- [ ] **1.1 Types.** `datasets/types.py` — `Doc` (`doc_id`, `title`, `text`,
  `.content` = title + text), `Query`, `Qrels = dict[str, dict[str, int]]`.
  No network imports: the metric tests construct these directly.

- [ ] **1.2 Download.** `datasets/nfcorpus.py` → `download_nfcorpus()`.
  Fetch `nfcorpus.zip`, verify hash, extract, write `.manifest.json`
  (url, sha256, counts). Idempotent; `--force` refetches.

- [ ] **1.3 Load.** Same file → `load_nfcorpus()` returning
  `(corpus, queries, qrels)` from `corpus.jsonl`, `queries.jsonl`,
  `qrels/test.tsv`. Test split only.
  *Done when:* a test asserts 3,633 docs / 323 queries and non-empty qrels.

- [ ] **1.4 Protocol.** `rag_core/retriever/base.py` — `SearchResult`
  (`doc_id`, `document`, `score`, `score_type`) and the `Retriever` protocol
  (`search(query, k) -> list[SearchResult]`). Types only, no behaviour.

- [ ] **1.5 BM25.** `rag_core/retriever/bm25.py` — `BM25Retriever` over
  `rank_bm25`, in-memory, `k1`/`b` configurable. **Needs dependency approval:
  `rank_bm25`.** Lowercase word-split tokenizer, deliberately untuned.

- [ ] **1.6 nDCG.** `metrics.py` (local to this package for now; moves to
  `rag_core.evals` in phase 3 once dense also uses it) — `ndcg_at_k` over
  ranked `doc_id`s and graded qrels. Unjudged = 0.
  *Done when:* unit tests match hand-computed nDCG on a toy 5-doc ranking.

- [ ] **1.7 Runner + CLI.** `evaluator.py` runs a `Retriever` over all queries
  → per-query nDCG + mean. `cli.py` with `download`, `run --experiment bm25`,
  `--limit N` for a fast smoke run.

- [ ] **1.8 Run JSON.** Write `results/runs/<experiment>_<ts>.json`:
  experiment, dataset, timestamp, config, metrics, `per_query`
  (`query_id`, `ndcg@10`, `latency_ms`, `retrieved` doc_ids), `llm_calls`.
  Document text never persisted.

- [ ] **1.9 GATE.** Run BM25 over all 323 queries.
  *Done when:* nDCG@10 is within ~0.02 of 0.32. If not, the fault is in
  1.2–1.6 — fix before starting phase 2.

**Deferred to later phases:** `benchmark.yaml` and `build_retriever` (phase 2,
when dense gives a second type to dispatch on), named resources (phase 2),
`git_commit` in the run JSON.

# tasks v3.0 — rag_bench_eval, phases 0–3

Retrieval-optimization benchmark on BEIR/NFCorpus. Design: `design.md`.
One task at a time; each phase ends runnable.

Phases 4–8 are in `design.md` §6 and get written up once phase 3 lands.

---

## Phase 0 — Scaffold

- [x] **0.1 Package skeleton.** `packages/rag_bench_eval/` with
  `pyproject.toml` (name `rag-optimizer-beir`, depends on `rag-core`),
  `src/rag_bench_eval/__init__.py`, `tests/`.
  *Done when:* `uv sync` succeeds and `import rag_bench_eval` works.

- [x] **0.2 Workspace registration.** Add to the root `pyproject.toml`:
  `[tool.uv.sources]` entry and `[tool.pytest.ini_options] testpaths`.
  *Done when:* `uv run pytest` from the root collects this package's tests.

- [x] **0.3 Settings.** `settings.py` with the dataset cache dir
  (`data/beir/nfcorpus/`) and results dir. Add `data/beir/` to `.gitignore` —
  NFCorpus is CC BY-SA, fetched at run time, never committed.

---

## Phase 1 — End-to-end: corpus → BM25 → nDCG@10

The gate. BM25 near the published nDCG@10 ≈ 0.32 validates dataset parsing,
qrels handling and the metric against one external number. Nothing later is
trustworthy until it does, so build the thinnest path that produces it.

- [x] **1.1 Types.** `datasets/types.py` — `Doc` (`doc_id`, `title`, `text`,
  `.content` = title + text), `Query`, `Qrels = dict[str, dict[str, int]]`.
  No network imports: the metric tests construct these directly.

- [x] **1.2 Download.** `datasets/nfcorpus.py` → `download_nfcorpus()`.
  Fetch `nfcorpus.zip`, verify hash, extract, write `.manifest.json`
  (url, sha256, counts). Idempotent; `--force` refetches.

- [x] **1.3 Load.** Same file → `load_nfcorpus()` returning
  `(corpus, queries, qrels)` from `corpus.jsonl`, `queries.jsonl`,
  `qrels/test.tsv`. Test split only.
  *Done when:* a test asserts 3,633 docs / 323 queries and non-empty qrels.

- [x] **1.4 Protocol.** `rag_core/retriever/base.py` — `SearchResult`
  (`doc_id`, `document`, `score`, `score_type`) and the `Retriever` protocol
  (`search(query, k) -> list[SearchResult]`). Types only, no behaviour.

- [x] **1.5 BM25.** `rag_core/retriever/bm25.py` — `BM25Retriever` over
  `rank_bm25`, in-memory, `k1`/`b` configurable. **Needs dependency approval:
  `rank_bm25`.** Lowercase word-split tokenizer, deliberately untuned.

- [x] **1.6 nDCG.** `metrics.py` (local to this package for now; moves to
  `rag_core.evals` in phase 3 once dense also uses it) — `ndcg_at_k` over
  ranked `doc_id`s and graded qrels. Unjudged = 0.
  *Done when:* unit tests match hand-computed nDCG on a toy 5-doc ranking.

- [x] **1.7 Runner + CLI.** `evaluator.py` runs a `Retriever` over all queries
  → per-query nDCG + mean. `cli.py` with `download`, `run --experiment bm25`,
  `--limit N` for a fast smoke run.

- [x] **1.8 Run JSON.** Write `results/runs/<experiment>_<ts>.json`:
  experiment, dataset, timestamp, config, metrics, `per_query`
  (`query_id`, `ndcg@10`, `latency_ms`, `retrieved` doc_ids), `llm_calls`.
  Document text never persisted.

- [x] **1.9 GATE.** Run BM25 over all 323 queries.
  *Done when:* nDCG@10 is within ~0.02 of 0.32. If not, the fault is in
  1.2–1.6 — fix before starting phase 2.

- [x] **1.10 CQADupStack Programmers dataset.** `datasets/cqadupstack_programmers.py` —
  `load_cqadupstack_programmers()` parses local BEIR-format files: `corpus.jsonl`,
  `queries.jsonl`, `qrels/test.tsv` under `data/cqadupstack/programmers/`. Adds
  setting `CQADUPSTACK_PROGRAMMERS_DIR` to `settings.py`.
  *Done when:* end-to-end run: corpus → BM25 → nDCG@10 over all CQADupStack
  Programmers queries; result recorded in `results/runs/` (JSON's `"dataset"`
  field identifies it as `cqadupstack_programmers`).
  Result: nDCG@10 = 0.2680 over 876 queries.

**Deferred to later phases:** `benchmark.yaml` and `build_retriever` (phase 2,
when dense gives a second type to dispatch on), named resources (phase 2),
`git_commit` in the run JSON.

---

## Phase 2 — `embeddings:` map, `DenseRetriever`, index cache → experiment 2

Dense is the second retriever type, so this is where config stops being CLI
constants: `benchmark.yaml`, the `embeddings:` resource map and
`build_retriever` all arrive because there is now something to dispatch on.
Embedding 3,633 docs is the one slow step in a sweep — the index cache exists
so it happens once, not once per experiment.

- [x] **2.1 `benchmark.yaml`.** Root of the package. `embeddings:` map,
  `retrieval.pipelines` with `bm25` and `dense`, `sweep`, `evaluation`
  (per `design_summary.md` §Config). Move `BM25_K1`/`BM25_B` out of `cli.py`.
  *Done when:* the phase 1 BM25 number reproduces from config, unchanged.

- [x] **2.2 Config loader.** `config.py` — read the YAML, return plain dicts.
  Add `PyYAML` to `pyproject.toml` (already installed transitively; declare it).
  No schema layer: `build_retriever` raises on an unknown `type`, and a
  missing resource name raises at lookup.

- [x] **2.3 `embeddings:` resources.** `resources.py` —
  `get_embeddings(name, cfg)`, a dict cache over `rag_core.embeddings.factory.
  get_embedding()` (provider dispatch already exists there; do not duplicate).
  Accept a bare name now; `{query:, passage:}` for asymmetric encoders is
  designed-for but not implemented (`design.md` §8 Q4).
  *Done when:* two lookups of one name return the same instance.

- [x] **2.4 `DenseRetriever`.** `rag_core/retriever/dense.py` — embed the
  corpus once, cosine over a `numpy` matrix, `score_type="cosine"`. Declare
  `numpy` explicitly in `rag_core`. No chunking: NFCorpus qrels are
  document-level, so `Doc.content` embeds whole (`design.md` §8 Q5).

- [x] **2.5 Index cache.** Cache the built index on its settings
  (embeddings name + metric for dense; `k1`/`b` for BM25) so one sweep embeds
  the corpus once. Runtime only — never in `benchmark.yaml`.
  *Done when:* a two-experiment sweep logs one embedding pass, not two.

- [x] **2.6 Embedding persistence.** Cache vectors to disk under
  `data/beir/nfcorpus/.embeddings/<model>.npy` keyed by model + corpus hash,
  so a re-run costs nothing. Gitignored with the rest of `data/beir/`.

- [x] **2.7 `build_retriever`.** `rag_core/retriever/factory.py` — recursive
  `if cfg["type"] == ...` dispatch over `(cfg, resources)`, returning bm25 or
  dense. Unknown type raises at load, naming the type and the valid set.
  *Done when:* a unit test builds both from dict configs.

- [x] **2.8 CLI sweep.** `run --experiment <name>` resolves from
  `benchmark.yaml`; add `--all` to run the `sweep` list and `list` to print
  available pipelines. Drop the hardcoded `bm25`-only branch.

- [x] **2.9 Experiment 2.** Run dense over all 323 queries; write its run JSON.
  *Done when:* nDCG@10 is recorded and sits in a plausible band (~0.30–0.35
  for a strong model). Unlike phase 1 there is no published number to match —
  a wildly low score means a bug, not a finding. Note the model used.

**Decision needed at 2.1:** dense baseline model — local `nomic-embed-text`
(free, committed default) or OpenAI `text-embedding-3-small` (~$0.01/corpus,
comparable to published baselines). `design.md` §8 Q2 suggests both: OpenAI for
the headline, Ollama as the default.

---

## Phase 3 — Extract the IR runner into `rag_core.evals`; results table

Two retrievers now share one runner, so its generic shape is visible rather
than guessed (`design.md` §6). The extraction is a move, not a rewrite: phase 2
numbers must reproduce byte-for-byte afterwards.

- [x] **3.1 Move metrics.** `metrics.py` → `rag_core/evals/metrics.py`,
  unchanged, with its tests. Drop the `Qrels` import from `rag_bench_eval` —
  take a plain `dict[str, int]` so `rag_core` keeps no dataset dependency.
  *Done when:* `uv run pytest` is green and the hand-computed nDCG tests pass
  from their new home.

- [x] **3.2 Add the remaining metrics.** `recall@100`, `mrr@10`,
  `precision@10` alongside `ndcg_at_k`, each with hand-computed unit tests
  (CLAUDE.md: metrics get real tests — a wrong metric invalidates every number).

- [x] **3.3 Extract the runner.** `evaluator.py` → `rag_core/evals/
  runner.py. Generalize: take `queries`/`qrels` as plain
  dicts, metric list from config, `dataset` as a parameter.
  *Done when:* re-running experiment 1 reproduces the phase 1 nDCG exactly.

- [x] **3.4 Multi-metric run JSON.** `metrics` carries all four scores;
  `per_query` keeps `ndcg@10` only (the diagnostic one) plus `latency_ms` and
  `retrieved`. Existing run files stay readable — additive keys only.

- [x] **3.5 Results table.** `report.py` — read `results/runs/*.json`, take the
  latest run per experiment, emit a markdown table (experiment, the four
  metrics, mean latency, `llm_calls`) sorted by nDCG@10. Wire `report` into
  the CLI, writing `results/results.md`.
  *Done when:* the table shows bm25 and dense side by side.

- [x] **3.6 Backfill.** Re-run experiments 1 and 2 through the extracted
  runner so both have four-metric runs, then regenerate `results.md`.

---

## Phase 4 — `RRFRetriever`, `RerankingRetriever` + `rerankers:` → experiments 5, 6, hybrid

Composition arrives before query expansion because fusion and reranking need no
LLM (`design.md` §6) — three more data points for zero tokens. This is also the
first phase where retrievers *nest*, so the two rules that produce wrong numbers
rather than errors (`design.md` §3.2) get enforced here: a parent asks its child
for the depth it needs, and `min_score` stays on the outermost node only.

- [x] **4.1 `RRFRetriever`.** `rag_core/retriever/fusion.py` — reciprocal rank
  fusion over n retrievers, `score = Σ 1/(rrf_k + rank)`, `score_type="rrf"`.
  Ranks are 1-based and per-child; a doc missing from a child contributes
  nothing. **Depth rule:** ask each child for at least the `k` requested, so
  fusing children that each return 10 can still yield a sensible top-10.
  *Done when:* a unit test over two hand-built rankings matches hand-computed
  RRF scores, including a doc found by only one child.

- [x] **4.2 `rerankers:` resources.** Add the map to `benchmark.yaml`
  (`bi_encoder` → `provider: bi_encoder` + an `embeddings:` reference;
  `cross_encoder` → `provider: cross_encoder` + `model:`) and
  `get_reranker(name, cfg)` to `resources.py`, dict-cached like
  `get_embeddings`. `provider` is the discriminator, not `type` —
  in `pipelines:` `type` means topology (`design.md` §3.3).

- [x] **4.3 Bi-encoder reranker.** Add to `rag_core/retriever/rerank.py` —
  re-scores candidates with an embedding model, cosine over query vs. candidate.
  Note: pointed at the same `default` the dense retriever uses it should show
  near-zero gain; that is the intended sanity check, not a bug (`design.md` §3.3).

- [x] **4.4 `RerankingRetriever`.** Same file — wraps an inner retriever, asks
  it for `candidate_k`, re-scores, returns `top_k`. `score_type="rerank_logit"`.
  The existing `CrossEncoderReranker`/`CohereReranker` operate on `Document`
  lists; reuse their `_score()` rather than duplicating provider logic, and
  adapt at the `SearchResult` boundary. **`candidate_k` must exceed `top_k`** or
  reranking has nothing to reorder.

- [x] **4.5 Extend `build_retriever`.** Add `rrf` (recurses over
  `cfg["retrievers"]`) and `rerank` (recurses into `cfg["inner"]`) to the
  dispatcher; extend `_VALID_TYPES`. Resources gains `get_reranker(name)`.
  *Done when:* a unit test builds the nested `hybrid_cross_encoder` config
  (rerank → rrf → [bm25, dense]) from a dict.

- [x] **4.6 Depth-rule test.** Over that nested config, assert the inner RRF
  actually receives `candidate_k=50` — not the 10 its children configure
  (`design.md` §7: a wrong `top_k`/`candidate_k` interaction degrades results
  silently). This is the one interaction with no external number to catch it.

- [x] **4.7 Index cache for composites.** Extend `build_pipeline_retriever`
  in `retrievers.py` so nested pipelines still hit the phase 2 caches — the
  bm25 and dense leaves inside `hybrid` must reuse cached indexes, not rebuild.
  *Done when:* a sweep of `dense` + `hybrid` embeds the corpus once.

- [x] **4.8 Pipelines + sweep.** Add `hybrid` (RRF over bm25 + dense),
  `bi_encoder_rerank` and `hybrid_cross_encoder` to `benchmark.yaml`; extend
  the `sweep` list.

- [x] **4.9 Experiments 5, 6, hybrid.** Run all three over 323 queries.
  *Done when:* `results.md` shows five experiments. Expect hybrid > either
  parent (lexical and dense fail differently) and cross-encoder > bi-encoder.
  A hybrid *below* both parents means the depth rule is broken — check 4.6
  before believing it as a finding.

**Dependency:** the cross-encoder needs `sentence-transformers`, an existing
`rag-core[rerank]` extra — declare it in `rag_bench_eval`, no new package.
First run downloads the model (~90 MB).

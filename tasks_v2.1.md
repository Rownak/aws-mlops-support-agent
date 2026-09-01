# Tasks — v2.1 rag_core rebuild on the ragwire architecture

Reference: `references/ragwire/` (+ `ragwire_design.md`). Branch: `v2.1-rag-pipeline-refactor`.
Retrospective log — every task below is already complete.

Legend: `[x]` done

---

## Phase 1 — Structural replacement

- [x] **1.1 Analyse the reference and plan the refactor**
  Read `references/ragwire/` end to end; wrote `ragwire_design.md` (architecture + the design
  ideas worth adopting) and `refactoring_tasks.md` (what to adopt, what NOT to copy and why,
  target structure, prioritised plan). Decision taken: full replacement, not incremental adoption.

- [x] **1.2 Replace rag_core with ragwire's structure**
  Ported `sources/`, `loaders/`, `processing/`, `embeddings/`, `retriever/`, `generation/`,
  `utils/` and the dot-notation `Config`. Added `llm/factory.py` — the reference's config declares
  an `llm:` block but ships no chat-model factory to match. Deleted as superseded: `chunking/`,
  `sources.py`, `generation/prompts.py`, `retrieval/retriever.py`.

- [x] **1.3 Keep Pinecone and the confidence heuristic**
  Rewrote `vectorstores/pinecone_store.py` against `QdrantStore`'s method surface (hash-based
  ingest status, delete-by-source, collection lifecycle); Qdrant-only facet/payload-index methods
  have no equivalent and were not ported. Kept `confidence.py` from the old engine, adapted to
  `(document, score)` pairs.

- [x] **1.4 Config surface + dependencies + tests**
  `config.example.yaml` (Pinecone + Ollama shaped, provider-switchable). `pinecone`,
  `langchain-pinecone`, `markitdown` become required; `ollama`/`openai`/`google`/`s3`/`rerank`/
  `cohere`/`hybrid` optional extras. Test suite rewritten for the new shapes; `test_boundary.py`
  needed no changes and still passes.

- [x] **1.5 Consolidate `retrieval/` into `retriever/`**
  A package holding one module was redundant — moved `confidence.py` across and deleted
  `retrieval/`.

## Phase 2 — Restore rag_core's own strengths

- [x] **2.1 Typed config with env > yaml > default precedence**
  Added a frozen `RagConfig` tree (`EmbeddingConfig`, `LLMConfig`, `VectorStoreConfig`,
  `SplitterConfig`, `RetrieverConfig`, `GenerationConfig`, `SourceSpec`) over the raw `Config`
  loader, restoring per-field precedence and fail-fast validation that reports every missing
  secret at once. Secrets read from the environment only. `.as_dict()` keeps the factories
  dict-based, so nothing downstream needed changing.

- [x] **2.2 Function-level seams for agent control flow**
  Extracted `retriever/retrieve.py` as a standalone function (ported from the reference's
  `RAGWire.retrieve`, minus its metadata-filter machinery) and added
  `RagCore.retrieve_with_confidence()` — the hook an agent uses to escalate on a weak match
  instead of generating. `query()`/`aquery()` became thin compositions of those seams.

## Phase 3 — Ingestion

- [x] **3.1 `ingest_documents` / `ingest_directory`**
  Built on a prepare/write split so one function owns every mutation and rollback has a single
  home. Content hash is the ingest identity: unchanged files skip, changed files replace their
  older version, a run that died mid-write is cleared and retried. Batched writes; per-file error
  capture so one bad file can't abort a run. Not ported: LLM metadata extraction, PageSplitter,
  Qdrant payload indexes, tqdm, threaded ingestion, the retry helper.

- [x] **3.2 Fold `sync()` into the same path**
  `sync()` now lists its sources and delegates to `ingest_documents`, so all three ingest entry
  points share one set of guarantees. Return type changed from an ad-hoc dict to `IngestStats`.

## Phase 4 — Structure and docs

- [x] **4.1 Split `config.py` into a `config/` package**
  447 lines in one file had a real defect: adding a provider meant edits in four disjoint places.
  Split into `loader.py`, `providers.py`, `pipeline_parts.py`, `env.py`, `base.py`, `root.py` (a
  clean DAG), with each block owning its `from_raw()` and `missing_secrets()`. Adding a provider
  is now a one-file edit. Public API unchanged — all names re-exported, so no consumer needed
  editing. Verified byte-identical `describe()`/`as_dict()` output against a pre-refactor snapshot.

- [x] **4.2 `RagCore.retrieve()` for inspecting raw retrieval**
  `retrieve_with_confidence()` discards the scores, so there was no way to see what retrieval
  actually ranked. Added a method returning `(document, score)` pairs; rewired
  `retrieve_with_confidence()` through it so the query path is one ladder.
  *(Superseded by 5.1: the scored shape moved to `retrieve_scored()` and `retrieve()` became
  the plain-document primitive.)*

- [x] **4.3 Rewrite `packages/rag_core/README.md`**
  Config precedence and env vars, the ingestion table, corrected layout tree, and the design notes
  that now carry the real insight (one-file provider adds, content-hash ingest identity).

## Phase 5 — Generic enough for any RAG app

The engine still assumed one app's shape in three places: retrieval could only speak in
similarity scores, a "ready" pipeline meant whatever the AWS project happened to have running,
and retrieval was invisible to tracing. Each is a thing a second corpus would have hit
immediately.

- [x] **5.1 Split `retrieve()` from `retrieve_scored()`**
  One function returning `(document, score)` forced every caller to care about scores, and had
  already cost a retrieval strategy — `retrieve()` raised outright for `search_type: mmr`, since
  MMR ranks for diversity and has no scored variant. Split into `retrieve()` (plain documents,
  every search type including MMR) and `retrieve_scored()` (the pairs confidence and evals need),
  so neither changes shape on a flag. `retrieve()` stamps `metadata["score"]` so scores stay
  available for logging without being the contract. Both mirrored on the `RagCore` facade.
  Rejected: a `return_scores: bool` from config — a value that changes a function's *return type*
  based on a file on disk, which no caller can reason about locally.

- [x] **5.2 Normalize scores; make the confidence threshold optional**
  `similarity_search_with_score` returns the backend's native metric — cosine in `[-1,1]`,
  dot-product unbounded, Euclidean a *distance* where lower is better. So `min_top_score` was only
  ever correct for cosine-on-Pinecone, and on a distance metric would have inverted and escalated
  everything: a latent bug that only a second backend would have exposed. Switched to
  `similarity_search_with_relevance_scores` (normalized `[0,1]`, higher-is-better) and moved
  `DEFAULT_MIN_TOP_SCORE` 0.35 → 0.675 to match the new scale. `min_top_score: null` now disables
  the check entirely — the honest way for a corpus with no established threshold to opt out.
  Explicit `null` needed its own parser: `_env_or` coalesces falsy values and would have silently
  restored the default.

- [x] **5.3 Backend-native `filters` pass-through**
  Both retrieval functions now take `filters`, forwarded to the vector store untouched.
  Deliberately *not* a neutral filter DSL — Pinecone, Qdrant and Chroma each use different filter
  syntax, and inventing a translation layer against a single backend would be guessing at the
  abstraction. Pass-through plus a docstring is honest and costs nothing to replace later.

- [x] **5.4 Restore LangSmith tracing on retrieval**
  The Pinecone search is a plain method call, not a `Runnable`, so LangChain's auto-instrumentation
  never saw it and retrieval was absent from traces — a regression from the ragwire port, which had
  no LangSmith dependency. Re-added `@traceable` on `retrieve`/`retrieve_scored` (`run_type="retriever"`,
  so they render as retriever spans with document cards) and on `assess_confidence`, decorated in
  the modules they trace. `langsmith` became an explicit dependency, which `design.md` had specified
  all along.

- [x] **5.5 Readiness preflight before pipeline construction**
  A misconfigured local backend surfaced as a raw SDK exception deep in first use — an SSL error
  from Pinecone Local, or a 404 for an Ollama model that was never pulled. Added `check_readiness()`,
  called from `RagCore.__init__` after config load and before any client is built: it verifies the
  configured Ollama models exist and that Pinecone Local answers, then raises one `RuntimeError`
  naming every problem and the exact command that fixes it. Kept out of `load_config()` on purpose —
  config parsing stays pure and offline-testable; live I/O belongs at construction. Skippable via
  `RAG_SKIP_READINESS_CHECK` for CI. Cloud providers are not probed: credentials are already
  `missing_secrets()`' job.

---

## Known issues, deliberately left

- `_env_or` uses `or`, so a YAML `top_k: 0` falls through to the default. Pre-existing, documented
  in `config/env.py`; fixing it belongs in its own change, not hidden inside a restructure.
- `packages/aws_mlops_support_agent/` is still written against the pre-v2.1 rag_core API
  (`ChunkingConfig`, `RetrievalConfig`, `make_retriever`, `SourceSpec.id`) and does not import.
  Out of scope by instruction — to be updated separately against the final shape. Phase 5 widened
  the gap (`retrieve()` changed shape), but did not create it, and the port is a rewrite rather
  than a signature fix.
- With a reranker configured, `retrieve_scored()` returns the reranker's relevance score — a
  cross-encoder logit on its own scale — while `min_top_score` still compares it as though it were
  a normalized similarity. Documented with a warning in the docstring rather than fixed: the fix
  wants a separate `min_rerank_score`, which is its own change.
- Eval baselines in `evals/results.md` predate the score normalization in 5.2 and need re-running;
  the old numbers are on a scale that no longer exists.

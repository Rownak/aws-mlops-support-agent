# tasks v2.2 — aws_mlops_support_agent migration onto the rearchitected rag_core

rag_core became a generic engine: a `RagCore` facade, `type`-dispatched
`Source` subclasses, and a new config schema. This package depended on all of
the old shapes, so every seam had to move. Two follow-on phases then pushed
work that had been patched around in this package back down into the engine.

---

## Phase 1 — Migrate aws_mlops_support_agent onto the new rag_core

- [x] **1.1 Config schema.** Rewrote `config.yml` from the old
  `index`/`models`/`chunking`/`retrieval` blocks into rag_core's generic
  schema (`embeddings`, `llm`, `vectorstore`, `splitter`, `retriever`,
  `generation`, `sources`). Decided **one config file per project**, not two:
  project-only settings (Jira, `DRY_RUN`) stay env vars read in
  `settings.py`, matching the `finance_report_rag` precedent.

- [x] **1.2 Ingestion path.** `DocSource`/`LoadedDoc` text-streaming and the
  free `run_ingest()` were gone. Reworked the awsdocs git-history recovery to
  write recovered markdown to local folders that `config.yml`'s sources point
  at, with a `_manifest.json` sidecar carrying each file's docs URL (rag_core
  had no field for it). `ingest.py` became fetch → `RagCore.sync()`.
  Clone root derived from `config.yml` rather than hardcoded.

- [x] **1.3 Agent/query path.** Replaced the removed `RetrievedChunk`,
  `make_retriever`, `generate_answer` and `AnswerPrompts`: chunks are now
  `(Document, score)` tuples from `RagCore.retrieve_scored()`; answers come
  from `AnswerGenerator` built with this project's system prompt; citations
  read chunk metadata instead of `.heading`/`.url`. Fixed the stale
  `rag_core.retrieval.*` import paths and the checkpointer's msgpack
  allowlist. Verified with real graph runs (retrieve → answer → interrupt →
  resume, plus the escalation path) using injected fakes.

- [x] **1.4 Evals.** Retrieval goes through `RagCore.retrieve_scored()`.
  Found and fixed a real mismatch: chunk labels are now full OS-specific file
  paths, so `expected_files` moved to plain filenames matched via a
  basename-rewriting retriever wrapper — portable across Windows/CI.

**Not in scope, still outstanding:** `packages/aws_mlops_support_agent/tests/`
— `conftest.py` still builds a `RagConfig` from the pre-migration flat schema
(`ChunkingConfig`, `RetrievalConfig`, `openai_api_key`, …), which blocks
collection of four test files.

---

## Phase 2 — Push the workarounds back into rag_core

- [x] **2.1 Move chunking out of the facade.** `_build_chunks` and
  `_splitter` were `RagCore` methods that never used `self`, so chunking
  couldn't be tested without a readiness check and a Pinecone connection.
  Both moved to `processing/chunking.py` as free functions (`build_chunks`,
  `splitter_from_config`). Kept the folder name `processing/` rather than
  `chunking/`: `hashing.py` is file-level dedup hashing, which the narrower
  name would mislabel.

- [x] **2.2 Generic per-file chunk metadata.** Added
  `Source.metadata_for(path) -> dict`, an optional override merged into every
  chunk of that file. Chose a generic metadata dict over a URL-specific hook,
  so any source can attach anything (URL, author, timestamps) — and it sets
  up metadata-filtered retrieval later. `build_chunks` refuses keys colliding
  with `RESERVED_METADATA_KEYS` rather than silently overwriting them (they
  drive dedup and partial-ingest detection); a collision fails that one file,
  not the run. `sync()` collects it per source; `ingest_documents` takes it
  keyword-only, so `ingest_directory` and existing callers are untouched.

- [x] **2.3 awsdocs as a real source type.** `AwsDocsGitSource` became a
  registered `Source` (`type: awsdocs_git`) instead of a pre-ingest script:
  `list_files()` owns clone → checkout pre-archival commit → strip anchors,
  and `metadata_for()` supplies the docs URL. That removed the reason the
  sidecar existed — the manifest, its cache and `doc_url_for()` are gone, and
  `ticket.py` reads `metadata["url"]` straight off a chunk, making it pure
  again. `ingest.py` collapsed to a single `RagCore.sync()` call.
  Rewrote `test_fetch_source.py`, which targeted an API two rewrites old.

- [x] **2.4 Move `AWSDOCS_REPOS` out of code into `config.yml`.** `git_url` and
  `docs_base_url` were hardcoded in a module-level dict in `sources/fetch.py`,
  so `config.yml` said *which* guide to ingest while Python said *what that
  guide is*. Adding one meant editing two files in lockstep — and uncommenting
  only the `sources:` entry failed at runtime with `Unknown awsdocs source id`.
  `AWSDOCS_REPOS` is gone; `AwsDocsGitSource.__init__` now takes `git_url` and
  `docs_base_url` directly, and CodePipeline is enabled in `config.yml`
  alongside CodeBuild with both keys set. No rag_core change needed —
  `build_source()` already forwards every unmatched config key as a kwarg.

  Also dropped `test_unknown_id_fails_before_any_cloning`: `id` validation
  against a hardcoded set no longer exists, since `id` is now just a label —
  any string is valid as long as `git_url`/`docs_base_url` are supplied.
  Touched `sources/fetch.py`, `config.yml` (both entries), `README.md`'s
  "adding a guide" snippet, and `test_fetch_source.py`'s constructor calls.

---

## Phase 3 — Documentation

- [x] **3.1 Package README.** Corrected everything the migration invalidated:
  the `LOADERS`/`loader:`/`DocSource` wiring, the old config schema names (in
  two places), the layout tree, and the ingest command. Added the
  `metadata_for` → chunk-URL flow and a config snippet for adding a guide.
  Replaced the "56 tests, offline" claim with an explicit *Known breakage*
  note for the `conftest.py` failure above, rather than quietly dropping the
  number.

---

## Phase 4 — Fix chunk sizing, split ingestion out of the facade

- [x] **4.1 Size chunks in tokens, not characters.** The splitters counted with
  `length_function=len`, so a configured `chunk_size: 800` meant 800
  *characters* (~200 tokens) — roughly a 4x shrink against the intended budget,
  which is what cut AWS procedure sections mid-step. All three factories now
  share one `_token_splitter` built with `from_tiktoken_encoder`
  (`cl100k_base`), so one configured number means the same thing whichever
  strategy a config picks. Three related bugs went with it: `keep_separator`
  was `False`, which deleted the heading text itself from each chunk (the
  separator *is* the `### `); `\n\n# ` was missing, so h1 sections were never
  break candidates; and `add_start_index` was computed then discarded, since
  `build_chunks` calls `split_text()`. Declared `tiktoken` explicitly — it was
  only present transitively via `langchain-openai`, so an ollama-only install
  would have failed at import.

- [x] **4.2 Extract `ingestion/` from the `RagCore` facade.** `pipeline.py`
  496 → 235 lines. `ingestion/ingestor.py` (`Ingestor`) owns the prepare →
  write path and the three ingest verbs; `ingestion/stats.py` holds
  `IngestStats`/`IngestError`/`empty_stats`. Deliberately a **pure move, no
  behavior change**, sequenced *before* the chunk-ID policy work: id format,
  reserved metadata keys, dedup, partial-ingest detection and stale-version
  deletion were spread across chunking, the store and the pipeline, so
  changing the ID policy touched all of them. `RagCore` keeps the three verbs
  as delegations (callers unchanged) and exposes `batch_size` as a property
  backed by the ingestor so the two cannot drift. `test_ingest.py` needed real
  edits, not just a move: its monkeypatch targets named a namespace the
  symbols no longer resolve in, and the fixture had to build an `Ingestor`
  rather than only setting `.store`/`.loader`.

---

## Phase 5 — Streamlit demo: ingest from the UI

- [x] **5.1 Add an ingest control to `demo/streamlit_app.py`.** The demo could
  only query — nothing in it reached `RagCore.sync()`, so a fresh Pinecone
  index (no prior `uv run aws-agent-ingest`) made every question fail inside
  `retrieve_scored` with no explanation in the UI. Added a sidebar button
  (`render_ingest_control`) that builds a `RagCore` from this project's
  `config.yml` (same `CONFIG_PATH` `nodes.py` already uses), calls `.sync()`,
  and renders the returned `IngestStats` via `render_ingest_summary`
  (processed/skipped/failed/replaced/chunks_created, plus any per-file errors
  in an expander) — mirroring what `ingest.py`'s CLI already prints, just in
  the sidebar. Runs inside `st.spinner`, since `sync()` is slow (git clone,
  chunk, embed) and blocking.

  `RagCore` is imported lazily inside the click handler, matching
  `nodes.py:60`'s reasoning: building it opens a Pinecone connection, which
  should not happen just from importing or rendering the page. Both new
  render functions are called from inside `with st.sidebar:`, since
  `st.sidebar.button(...)` only scopes the button itself — a bare `st.success`
  called afterward renders to the main area, not the sidebar.

  Added `render_ingest_summary` to `test_demo.py`'s existing pure-function
  coverage (Streamlit calls are no-ops in bare mode, so these just check the
  function runs on a real `IngestStats` shape without raising).


---

## Phase 6 — Move the eval runner out of rag_core

`rag_core/evals/runner.py` takes its router as an injected callable specifically
so it doesn't depend on any particular agent — but in practice it only has one
caller (`aws_mlops_support_agent/evals/run.py`), and `should_escalate`/`EvalCase`
are shaped around this agent's retrieve → confidence → escalate flow. That
makes it project-specific code sitting in the generic engine package, the same
shape Phase 2 fixed for chunking. Nothing else in `rag_core` imports it.

- [x] **6.1 Move `evals/runner.py` and its test into this package.**
  `rag_core/src/rag_core/evals/runner.py` →
  `aws_mlops_support_agent/src/aws_mlops_support_agent/evals/runner.py`;
  `rag_core/tests/test_evals.py` → this package's `tests/`. Pure move, no
  behavior change. Updated imports in `evals/run.py` and `evals/dataset.py`
  from `rag_core.evals.runner` to `aws_mlops_support_agent.evals.runner`, plus
  the stale docstring mentions of the old path in both files and in
  `runner.py`'s own module docstring. Left `rag_core/src/rag_core/evals/`
  in place (just `__init__.py`) rather than deleting it, per instruction —
  `rag_bench_eval`'s tasks_v3.0 has its own, unrelated plan to land a generic
  IR-metrics runner at that same `rag_core.evals` path later, so the package
  stays.

- [x] **6.2 Verify.** `rag_core`: `uv run pytest` → 175 passed. The moved
  and updated modules (`evals/runner.py`, `evals/dataset.py`, `evals/run.py`)
  import cleanly and `test_evals.py`'s 9 tests pass when run directly.
  Collecting `aws_mlops_support_agent`'s full `tests/` via pytest still fails
  at `conftest.py` — the pre-existing, already-documented breakage from Phase
  1 (`ChunkingConfig`/`RetrievalConfig` import from the old flat schema),
  reproduced as unchanged before touching anything eval-related, so it's not
  a regression from this move. Did not run `uv run aws-agent-evals` against
  a live index (needs a populated Pinecone index + API keys).

---

## Notes / open items

- **CodePipeline is now enabled** in `config.yml` alongside CodeBuild (2.4).
  The plan (see the fresh-index note below) is to ingest CodeBuild first,
  confirm it, then bring CodePipeline in — `sync()` already skips unchanged
  files by content hash, so re-running after enabling a new source only
  embeds what's new. Once both are live, revisit the README's "two awsdocs
  sources" wording and the 5 CodePipeline eval cases in `evals/dataset.py`.
- **The existing Pinecone index is stale after 4.1.** Every stored chunk was
  embedded under character-sized splitting. The *files* did not change, so
  `file_hash` is unchanged and `get_ingest_status` reports `complete` — a
  re-ingest skips everything and will not fix it. The collection has to be
  cleared, which is entangled with the delete/sync semantics below.
- **Deferred, discussed but not started:** (a) chunk-ID policy and delete/sync
  semantics — `f"{file_hash}_{i}"` is idempotent only for *unchanged* files,
  since an edit re-ids every chunk; note `aws_mlops_support_agent/README.md`
  still claims "Idempotent (deterministic chunk IDs)", left as-is pending that
  decision. Switching to position-keyed ids (`source#index`) would break
  `get_ingest_status`, which relies on `file_hash` to tell complete from
  partial. (b) Restoring the real header-splitter stage — 4.1 makes headings
  *preferred break points*, but the old two-stage `MarkdownHeaderTextSplitter`
  also extracted `heading` metadata (`h1 > h2 > h3`), which nothing carries
  now; that changes the `splitter_from_config` → `build_chunks` contract, so
  it is cleanest after the identity work settles.
- **Pydantic** was discussed and deliberately not adopted: worthwhile for the
  *new* query-side metadata extraction (structured LLM output), a low-risk
  mechanical swap for the config tree, but not itself a solution for
  filterable chunk metadata — that needs a schema threaded through
  `build_chunks`/`metadata_for` regardless of container type.

  Errors: Failed to process data\aws_docs\codebuild\doc_source\build-spec-ref.md: File conversion failed after 1 attempts:
 - PlainTextConverter threw UnicodeDecodeError with message: 'ascii' codec can't decode byte 0xe2 in position 32238: ordinal not in range(128)

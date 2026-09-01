# Tasks — v2 generic RAG workspace

Design: `design.md`. Branch: `v2-generic-rag-pipeline`.
**One task at a time.** Each task ends with a green `uv run pytest` and a note in `progress.md`.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## Phase 0 — Workspace scaffold

- [x] **0.1 Create the uv workspace skeleton**
  Add `[tool.uv.workspace] members = ["packages/*"]` to the root `pyproject.toml`; create
  `packages/rag_core/` and `packages/aws_mlops_support_agent/` with their own `pyproject.toml`
  (`rag-core` and `aws-mlops-support-agent`), `src/<pkg>/__init__.py`, and empty `tests/`.
  Split dependencies per design §7. Keep `src/` untouched and working.
  *Done when:* `uv sync` resolves the workspace and `uv run pytest` still passes against `src/`.

- [x] **0.2 Move shared tooling config to the root**
  Ruff settings and pytest config at the workspace root, covering both packages.
  *Done when:* `uv run ruff check .` passes across the repo.

## Phase 1 — Foundations (config + logging)

- [x] **1.1 Port config to `rag_core.config`**
  Move the generic half of `src/config.py` (OpenAI, Pinecone, index, region) to
  `rag_core/config.py` as a frozen `RagConfig`. Env-only for now — **no YAML yet**. Preserve
  fail-fast collect-all-missing behavior verbatim.
  *Done when:* ported config tests pass in `packages/rag_core/tests/`.

- [x] **1.2 Add `config.yml` loading + precedence**
  Layer YAML on top of 1.1: env var > `config.yml` > default. Add `pyyaml` to `rag-core`.
  Write `packages/aws_mlops_support_agent/src/aws_mlops_support_agent/config.yml` per design §5.
  *Done when:* tests cover all three precedence levels and a missing/malformed file fails clearly.

- [x] **1.3 Create `aws_mlops_support_agent.settings`**
  Project-side wrapper holding Jira vars + `DRY_RUN` around `RagConfig`. Port the fail-safe
  `_parse_dry_run` exactly (only explicit false/0/no disables dry-run).
  *Done when:* dry-run tests pass, including the "typo stays safe" case.

- [x] **1.4 Move `observability.py` into `rag_core`**
  *Done when:* `test_observability.py` passes from `packages/rag_core/tests/`.

## Phase 2 — Ingest engine

- [x] **2.1 Define the `DocSource` protocol + `SourceSpec`**
  `rag_core/sources.py` per design §6: what a corpus adapter must provide, no AWS knowledge.
  *Done when:* a fake in-memory source satisfies the protocol in a test.

- [x] **2.2 Move chunking to `rag_core/chunking/`**
  Port `src/ingest/chunk.py`. Make chunk size/overlap and the strip-patterns list config-driven
  (the awsdocs `<a name>` regex becomes one configured pattern, not a hardcoded constant).
  Rename the `service` metadata field to `source_id`.
  *Done when:* `test_ingest_chunk.py` passes with the AWS anchor pattern supplied via config.

- [x] **2.3 Move the vector store to `rag_core/vectorstore/`**
  Port `src/ingest/index.py`, keeping both the dimension guard and the create-vs-query split
  (query-time must still fail loudly rather than create an empty index).
  *Done when:* tests cover unknown-model, dimension-mismatch, and missing-index errors.

- [x] **2.4 Move the AWS fetcher to the project package**
  `src/ingest/fetch.py` → `aws_mlops_support_agent/sources/fetch.py`, implementing `DocSource`.
  Sources come from `config.yml`, not the hardcoded `REPOS` list.
  *Done when:* the adapter satisfies the protocol; git-history logic is unchanged.

- [x] **2.5 Write `rag_core.pipeline` (ingest path)**
  Orchestrate source → load → chunk → embed → upsert, driven entirely by config + injected sources.
  *Done when:* an end-to-end ingest test runs with a fake source and fake store, no network.

## Phase 3 — Query engine

- [x] **3.1 Move retrieval to `rag_core/retrieval/`**
  Port `retriever.py`; keep the injectable-store design and `RetrievedChunk` (with `source_id`).
  *Done when:* `test_retriever.py` passes.

- [x] **3.2 Move confidence to `rag_core/retrieval/`**
  Port `confidence.py`; `MIN_TOP_SCORE` becomes `retrieval.min_top_score` from config.
  *Done when:* `test_confidence.py` passes, including a non-default threshold case.

- [x] **3.3 Move generation to `rag_core/generation/`**
  Port `answer.py` + `prompts.py`. Prompts get a `rag_core` default that a project can override.
  *Done when:* `test_answer.py` passes and a custom prompt override is covered.

- [x] **3.4 Add the generic `rag-ask` CLI**
  Port `src/rag/ask.py` to `rag_core`, taking `--config` so any project can use it.
  *Done when:* `uv run rag-ask "…" --config <path>` returns a cited answer.

## Phase 4 — Evals

- [x] **4.1 Split the eval runner from the dataset**
  `EvalCase` + runner → `rag_core/evals/`; the 15 AWS questions + `results.md` →
  `aws_mlops_support_agent/evals/`.
  *Done when:* `test_evals.py` passes; the runner has no AWS-specific knowledge.

## Phase 5 — Agent & UI

- [x] **5.1 Move the agent package**
  `src/agent/*` → `aws_mlops_support_agent/agent/`, importing RAG functions from `rag_core`.
  *Done when:* `test_graph.py`, `test_routing.py`, `test_ticket.py`, `test_jira_tool.py` pass.

- [x] **5.2 Move the Streamlit demo**
  *Done when:* `test_demo.py` passes and the UI runs locally with Jira forced to dry-run.

- [x] **5.3 Wire console-script entrypoints**
  `aws-agent`, `aws-agent-ingest`, `aws-agent-evals`, `aws-agent-demo`, `rag-ask` (design §7).
  *Done when:* each command runs from a clean `uv sync`.

- [x] **5.4 Delete `src/`**
  Only once every module has a new home and the full suite is green.
  *Done when:* `src/` is gone, `uv run pytest` passes, no stale imports remain.

## Phase 6 — Boundary enforcement & deploy

- [x] **6.1 Add the dependency-direction check**
  CI check that `rag_core` never imports a project package (design §6). Also verify
  `rag_core`'s tests pass with only `rag-core` installed.
  *Done when:* the check fails on a deliberately added bad import, then passes once reverted.

- [x] **6.2 Update Dockerfile + `.dockerignore`**
  Build the workspace, install both packages, run the demo entrypoint.
  *Done when:* `docker build` succeeds and the container serves on 8501.

- [x] **6.3 Update GitHub Actions + ECS task definition**
  New paths, new entrypoint commands.
  *Done when:* CI is green and the ECS task definition references the new command.

## Phase 7 — Docs

- [x] **7.1 Rewrite `README.md` for the workspace**
  New project structure, new commands, a short "how to add a new RAG project" section.
  *Done when:* every command in the README has been run and works as written.

- [x] **7.2 Write `README.md` for each packages**
  Package structure, commands
  *Done when:* every command in the README has been run and works as written.


## Phase Future — Prove the seam

- [ ] **Future.1 Scaffold `packages/scifact_rag/`**
  Minimal second project: `config.yml`, a `DocSource` for the SciFact dataset, ingest + ask. **No**
  agent, **no** Jira, **no** UI — this exists to prove `rag_core` is genuinely reusable.
  *Done when:* it ingests and answers a question using only `rag_core` + its own source adapter.

- [ ] **Future.2 Fold back what Future.1 taught us**
  Anything `scifact_rag` had to work around becomes a `rag_core` fix. Expect a few — that is the
  design working, not failing.
  *Done when:* the workarounds are gone and both projects use `rag_core` unmodified.

# Design — Generic RAG workspace (v2)

Status: proposed · Branch: `v2-generic-rag-pipeline` · Supersedes the flat `src/` layout.

## 1. Goal

Turn a single-purpose project (`src/`, AWS-docs-only) into a **uv workspace** of independently
installable packages, so a second and third RAG project (`scifact_rag`, and later medical/financial
corpora) can be added without copying code and without cloning the whole repository.

Three requirements drive every decision below:

1. **Reuse** — one RAG engine, many corpora. Adding a corpus should mean writing config + a source
   adapter, not re-implementing chunking, embedding, retrieval, or evals.
2. **Independent installability** — `rag_core` must be installable on its own (`pip install
   ./packages/rag_core`, or from an index later), without dragging in LangGraph, Streamlit, or Jira.
3. **No regression** — the existing agent keeps working, with its 60+ offline tests still passing.

Non-goal for v2: publishing to PyPI. The packaging is *ready* for it; actually publishing is a later
decision.

## 2. Why refactor rather than rewrite

The current `src/` already separates ingest / rag / agent / demo / evals, and those seams map almost
1:1 onto the target split. The 60+ offline tests (fakes injected for retriever, LLM, and Jira) act as
the safety net during the move. A rewrite would discard working, tested, deployed code to arrive at
the same place with more risk and no reuse benefit.

## 3. Target layout

```text
aws-mlops-support-agent/              # repo root (rename deferred)
├── pyproject.toml                    # uv workspace root — members, shared dev deps, ruff config
├── uv.lock                           # single lockfile for the whole workspace
├── design.md
├── tasks.md
├── packages/
│   ├── rag_core/                     # generic, corpus-agnostic RAG engine
│   │   ├── pyproject.toml            # name = "rag-core"; no LangGraph / Streamlit / Jira deps
│   │   ├── src/rag_core/
│   │   │   ├── config.py             # RagConfig: config.yml + env overlay, secrets from env only
│   │   │   ├── pipeline.py           # orchestration: load → chunk → embed → upsert; and query path
│   │   │   ├── sources.py            # DocSource protocol + SourceSpec (what a corpus adapter must provide)
│   │   │   ├── loaders/              # raw bytes → Documents (markdown_loader, later markitdown_loader)
│   │   │   ├── chunking/             # Documents → chunks (markdown header split, token splitter)
│   │   │   ├── vectorstore/          # index.py — Pinecone create/verify/upsert, dimension guard
│   │   │   ├── retrieval/            # retriever.py (RetrievedChunk), confidence.py
│   │   │   ├── generation/           # answer.py, prompts.py, ask.py (single-question CLI)
│   │   │   ├── evals/                # EvalCase type + runner (dataset lives with each project)
│   │   │   └── observability.py      # JSON-lines logging (log_event)
│   │   └── tests/                    # engine tests, no project-specific fixtures
│   ├── aws_mlops_support_agent/      # this project: AWS docs corpus + support agent
│   │   ├── pyproject.toml            # depends on rag-core (workspace path dep)
│   │   ├── src/aws_mlops_support_agent/
│   │   │   ├── config.yml            # corpus + index + model settings for THIS project
│   │   │   ├── settings.py           # AgentConfig: Jira vars + DRY_RUN, wraps RagConfig
│   │   │   ├── sources/              # fetch.py — awsdocs git-history recovery, DocSource impl
│   │   │   ├── agent/                # LangGraph state machine, Jira tool, ticket builder
│   │   │   ├── demo/                 # Streamlit UI
│   │   │   ├── evals/                # this corpus's eval dataset + results.md
│   │   │   └── app.py                # CLI entrypoint
│   │   └── tests/                    # agent/graph/Jira/ticket tests
│   └── scifact_rag/                  # later — same shape, proves the seam
└── deploy/                           # Dockerfile targets + ECS runbooks (stays at root for now)
```

### Naming decisions (settled)

- **`rag_core`** is the name everywhere — directory, distribution (`rag-core`), and import root.
- **No inner `core/`** — `rag_core.config` / `rag_core.pipeline`, not `rag_core.core.config`.
- **`loaders/` and `chunking/` are separate** — a new corpus commonly swaps one without the other.
- **`retrieval/`** (not `retriever/`) for symmetry with `generation/` and `chunking/`.

## 4. The generic / project-specific boundary

This is the crux of the refactor. Each current module has exactly one home:

| Current | Goes to | Why |
|---|---|---|
| `src/ingest/chunk.py` | `rag_core/chunking/` | Markdown header + token splitting is corpus-agnostic. The awsdocs `<a name>` anchor regex is the one AWS-ism — it becomes a configurable cleanup rule. |
| `src/ingest/index.py` | `rag_core/vectorstore/` | Pinecone create/verify/upsert + dimension guard. Pure infrastructure. |
| `src/rag/retriever.py` | `rag_core/retrieval/` | `RetrievedChunk` + `retrieve()` already take an injectable store. |
| `src/rag/confidence.py` | `rag_core/retrieval/` | Heuristic is generic; `MIN_TOP_SCORE` becomes config (thresholds are corpus-dependent). |
| `src/rag/answer.py`, `prompts.py`, `ask.py` | `rag_core/generation/` | Prompts get a project-overridable default. |
| `src/observability.py` | `rag_core/` | Used by both layers. |
| `src/evals/run.py` | `rag_core/evals/` | Runner is generic… |
| `src/evals/dataset.py` | `aws_mlops_support_agent/evals/` | …but the *questions* are corpus-specific. `EvalCase` type stays in `rag_core`. |
| `src/ingest/fetch.py` | `aws_mlops_support_agent/sources/` | awsdocs git-history recovery is deeply AWS-specific. |
| `src/ingest/sources.py` | split | The `DocRepo` *shape* generalizes to `SourceSpec` in `rag_core`; the two AWS `REPOS` entries move to this project's `config.yml`. |
| `src/agent/*`, `src/demo/*`, `src/app.py` | `aws_mlops_support_agent/` | Agent, Jira, UI — all project-specific. |
| `src/config.py` | split | OpenAI/Pinecone/model/index settings → `rag_core.config`. Jira vars + `DRY_RUN` → this project's `settings.py`. |

**Rule of thumb:** if it mentions AWS, Jira, or CodeBuild, it belongs to the project package. If it
would be equally true of a medical-records corpus, it belongs to `rag_core`.

## 5. Configuration model

Two layers, deliberately different in kind:

- **`config.yml` (per project, committed)** — non-secret, corpus-shaped settings: index name,
  embedding + chat model, chunk size/overlap, top-k, confidence threshold, and the document sources.
  Committing it makes each project's corpus reproducible and diffable.
- **Environment variables (never committed)** — secrets only: `OPENAI_API_KEY`, `PINECONE_API_KEY`,
  Jira credentials. Plus optional overrides for the non-secret keys, so ECS/CI can change an index
  name without editing a file.

Precedence: **env var > config.yml > built-in default.** `rag_core.config.load_config(path)` returns
a frozen `RagConfig`; `aws_mlops_support_agent.settings` wraps it with the Jira/`DRY_RUN` fields it
alone needs. This preserves the current fail-fast behavior (collect all missing required vars, report
in one pass) and the fail-safe `DRY_RUN` parse (only an explicit false/0/no disables it).

Sketch of `aws_mlops_support_agent/config.yml`:

```yaml
project: aws-mlops-support-agent
index:
  name: aws-mlops-docs
  metric: cosine
models:
  embedding: text-embedding-3-small
  chat: gpt-4o-mini
chunking:
  size_tokens: 800
  overlap_tokens: 100
  strip_patterns: ['<a name="[^"]*"></a>']   # awsdocs heading anchors
retrieval:
  top_k: 4
  min_top_score: 0.35
sources:
  - id: codebuild
    loader: awsdocs_git
    git_url: https://github.com/awsdocs/aws-codebuild-user-guide.git
    docs_base_url: https://docs.aws.amazon.com/codebuild/latest/userguide/
  - id: codepipeline
    loader: awsdocs_git
    git_url: https://github.com/awsdocs/aws-codepipeline-user-guide.git
    docs_base_url: https://docs.aws.amazon.com/codepipeline/latest/userguide/
```

The `service` metadata field is renamed to the neutral `source_id` in chunk metadata, since "service"
only makes sense for AWS.

## 6. The extension seam

`rag_core` cannot know how to fetch every corpus. It defines a narrow protocol; each project
implements it:

```python
# rag_core/sources.py
class DocSource(Protocol):
    def fetch(self) -> Iterable[LoadedDoc]: ...   # raw docs + provenance metadata
```

`aws_mlops_support_agent/sources/fetch.py` implements the awsdocs git-history recovery behind that
protocol. `scifact_rag` will implement a dataset-download version. `rag_core.pipeline` consumes the
protocol and never imports a project package — **dependencies point one way only:
project → rag_core.** A CI import check enforces this.

## 7. Packaging & dependencies

Root `pyproject.toml` declares the workspace:

```toml
[tool.uv.workspace]
members = ["packages/*"]
```

`aws_mlops_support_agent` depends on `rag-core` as a workspace source (path dep locally; a version
pin if ever published). Dependency split:

- **`rag-core`**: langchain, langchain-openai, langchain-pinecone, langchain-text-splitters,
  pyyaml, python-dotenv, langsmith.
- **`aws_mlops_support_agent`**: rag-core, langgraph, streamlit, requests (Jira), grandalf.

This split is the payoff for requirement 2: installing `rag-core` alone pulls no agent framework, no
web UI, no Jira client.

Entrypoints move to console scripts so commands stay short and layout-independent:

| Before | After |
|---|---|
| `python -m src.ingest` | `uv run aws-agent-ingest` |
| `python -m src.app` | `uv run aws-agent` |
| `python -m src.evals` | `uv run aws-agent-evals` |
| `python -m src.rag.ask "…"` | `uv run rag-ask "…"` (generic, `--config` flag) |
| `streamlit run src/demo/streamlit_app.py` | `uv run aws-agent-demo` |

## 8. Testing strategy

Tests split by package and stay fully offline (fakes for retriever, LLM, Jira — no network, no keys):

- `packages/rag_core/tests/` — chunking, retrieval, confidence, config precedence, vectorstore
  guards. Must pass with `rag_core` installed **alone**; that is the real proof of independence.
- `packages/aws_mlops_support_agent/tests/` — graph routing, interrupts, ticket building, Jira tool,
  demo. Existing tests move here largely unchanged apart from imports.

`uv run pytest` from the root runs both. The migration is done incrementally, running the suite after
each move, so a break is always attributable to the step that caused it.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Import churn breaks things silently | Move one module at a time; run `uv run pytest` after each. Ruff catches unused/broken imports. |
| Config refactor changes runtime behavior | Port `load_config` semantics verbatim first (fail-fast list, fail-safe `DRY_RUN`), *then* layer YAML on top. Keep `test_config.py` green throughout. |
| Deploy breaks (Dockerfile/ECS paths) | Deploy changes land last, as their own task, after the suite is green locally. |
| `rag_core` accidentally imports the agent | Explicit CI check that `rag_core` never imports a project package. |
| Over-abstracting for corpora that don't exist yet | Generalize only what the AWS project actually needs now. `scifact_rag` (task 10) is the first real test of the seam — expect to adjust `rag_core` then, and treat that as the design working, not failing. |

## 10. Migration order

Bottom-up, so each step lands on already-moved foundations:

workspace scaffold → config → observability → loaders/chunking → vectorstore → retrieval →
generation → evals → agent/demo → entrypoints → deploy → docs → `scifact_rag` proof.

Detailed, one-at-a-time steps live in `tasks.md`.

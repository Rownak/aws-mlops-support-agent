# rag-core

**A corpus-agnostic retrieval-augmented-generation engine.** Point it at a corpus with a `config.yaml`
and it handles the rest: loading, chunking, embedding, indexing, retrieval, reranking, confidence
scoring, grounded answer generation, and evaluation.

It knows nothing about AWS, Jira, LangGraph, or Streamlit. That is enforced, not merely intended — see
[The boundary](#the-boundary) below.

Architecture is adopted from a reference project (`references/ragwire/`); Pinecone is the vector-store
backend, and `retriever/confidence.py` plus the typed `RagConfig` layer are rag-core-specific additions
on top of it.

---

## Install

Inside this workspace it's already installed by `uv sync` at the repo root. On its own:

```bash
pip install ./packages/rag_core
```

Provider-specific pieces are optional extras, so a minimal install stays light:

```bash
pip install "rag-core[ollama]"        # local LLM + embeddings
pip install "rag-core[openai]"        # cloud LLM + embeddings
pip install "rag-core[s3]"            # S3 document source
pip install "rag-core[rerank]"        # local cross-encoder reranking
pip install "rag-core[cohere]"        # hosted Cohere reranking
pip install "rag-core[hybrid]"        # dense + sparse retrieval
```

**Requires:** Python 3.13+. A `PINECONE_API_KEY` for the managed service, or `vectorstore.host` pointed
at Pinecone Local (Docker) for local dev with no key at all.

---

## Configuration

Everything is declared in one YAML file — see `config.example.yaml` for the annotated reference:

```yaml
embeddings:
  provider: "ollama"                  # or "openai" | "huggingface" | "google" | "fastembed"
  model: "nomic-embed-text"
  base_url: "http://localhost:11434"

llm:
  provider: "ollama"                  # or "openai" | "google"
  model: "llama3.1:8b"

vectorstore:
  provider: "pinecone"
  collection_name: "my-project-docs"
  # host: "http://localhost:5080"     # set to use Pinecone Local instead of the managed service

retriever:
  search_type: "similarity"           # "similarity" | "mmr" | "hybrid"
  top_k: 5
  min_top_score: 0.35                 # below this, treat retrieval as low-confidence

sources:
  - type: local
    path: "./documents"
    recursive: true
```

**Precedence is env var > `config.yaml` > built-in default**, per field. Secrets are read *only* from
the environment (`PINECONE_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `COHERE_API_KEY`) — never from
a value written in the YAML. `${VAR}` placeholders inside the file are resolved from the environment at
load time. Tuning values can also be overridden by env var: `RAG_TOP_K`, `RAG_MIN_TOP_SCORE`,
`RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `AWS_REGION`.

Missing secrets fail fast, and **all** of them are reported in one error rather than one per run. Only
what the configured provider actually needs is required: Ollama needs no key, and neither does a
Pinecone reached via `host`.

Because the LLM, embedding model, and vector store are all provider-switchable from config alone,
moving between local development (Ollama + Pinecone Local) and a managed deployment, or between corpora
(financial docs, medical docs, AWS docs), needs no code change.

---

## Usage

```python
from rag_core import RagCore

rag = RagCore("config.yaml")
rag.sync()                                    # reconcile the index against configured sources
answer = rag.query("How do I cache dependencies?")
print(answer.formatted())                     # answer text + numbered "Sources:" list
```

```bash
uv run rag-ask "your question" --config path/to/config.yaml
```

`RagCore` is a convenience facade. Every step it composes is also usable standalone —
`retriever.retrieve`, `retriever.confidence.assess_confidence`, and
`generation.generator.AnswerGenerator.generate`/`agenerate` — which is what lets an agent interleave
its own control flow. `rag.retrieve_with_confidence(question)` returns `(documents, confidence)` so a
caller can escalate on a weak match instead of generating an answer at all.

### Ingestion

| Method | What it does |
|---|---|
| `rag.ingest_documents([...])` | Ingest a specific list of files |
| `rag.ingest_directory(path, recursive=, extensions=)` | Ingest everything matching in a directory |
| `rag.sync()` | Ingest whatever the configured `sources` currently list |

All three share one code path and return the same `IngestStats` (`total`, `processed`, `skipped`,
`failed`, `chunks_created`, `replaced`, `errors`). Files are identified by **content hash**, so
re-running is free for unchanged files, a changed file replaces its older version, and a run that died
mid-write is cleared and retried rather than left truncated.

Every chunk carries provenance metadata (`source`, `file_name`, `file_hash`, `chunk_index`,
`total_chunks`, …). A custom `Source` can attach **extra** per-file metadata — a canonical URL to cite,
a field to filter retrieval on — by overriding `metadata_for(file_path)`; see the example in
`sources/__init__.py`. Those keys must not collide with the reserved ones
(`processing.chunking.RESERVED_METADATA_KEYS`), and a collision fails that one file rather than the run.

---

## Commands

| Command | What it does |
|---|---|
| `uv run rag-ask "…" --config <path>` | Retrieve → generate a cited `Answer`. `-k N` overrides `top_k`. |
| `uv run pytest packages/rag_core/tests` | The engine's own suite (142 tests, fully offline). |

---

## Layout

```text
src/rag_core/
├── config/              # loader.py (raw YAML + ${VAR}), providers.py, pipeline_parts.py,
│                        #   env.py (precedence), base.py, root.py (RagConfig, load_config, describe)
├── pipeline.py          # RagCore facade: ingest_documents / ingest_directory / sync / query / aquery
├── sources/             # Source ABC + lazy REGISTRY; local.py, s3.py — "which files exist now?"
├── loaders/             # bytes -> markdown text (MarkItDown: PDF/DOCX/XLSX/PPTX/...)
├── processing/          # hashing.py (content identity for ingest dedup), splitter.py (chunking)
├── embeddings/          # factory.py: provider string -> Embeddings
├── llm/                 # factory.py: provider string -> chat model
├── vectorstores/        # pinecone_store.py: index lifecycle + hash-based ingest-state tracking
├── retriever/           # retrieve.py, hybrid.py, rerank.py, confidence.py
├── generation/          # answer.py (Answer/Citation), generator.py (context budget, citations), ask.py
├── evals/               # EvalCase + hit@k and escalation-accuracy runner, markdown table
└── observability.py     # log_event(): one JSON object per line, CloudWatch-parsable
```

---

## Design notes

- **Config is a typed tree over a raw YAML loader.** `Config` reads the file; `RagConfig` is a frozen
  dataclass tree built on top of it, with per-field env>yaml>default precedence and fail-fast
  validation. Each block owns its own `from_raw()` and, where it carries a secret, `missing_secrets()`
  — so **adding a provider is a one-file edit** (`config/providers.py`), not four scattered ones.
- **Content hash is the unit of ingest identity**, not path or mtime. Each chunk carries a stamped
  `total_chunks`, giving a tri-state ingest status (absent / partial / complete). A failed write is
  rolled back, so a half-written file retries cleanly instead of looking done forever.
- **A registry with lazy imports.** `sources.base.REGISTRY` populates on first access and exposes
  `.register()`, so an optional dependency (boto3) never sits on the import path for a project that
  doesn't use it.
- **Retrieval is two-stage by construction.** `resolve_fetch_k` widens the candidate pool before an
  optional reranker narrows it back to `top_k`.
- **Refusal is a protocol, not a phrase.** `REFUSAL_SENTINEL` is detected by containment plus a length
  guard, so prompt wording can change without breaking refusal detection.
- **The answer is an object, not a string.** `Answer` carries text, `Citation`s, the retrieved set, and
  a `confidence` — citation coverage, which measures traceability, not correctness (see its docstring).
- **A context character budget is enforced by the library**, so the provider never silently truncates
  and drops the best-ranked chunk last.
- **`assess_confidence`** is an explainable score-threshold heuristic (`is_confident`, `score_gap`) an
  agent uses to decide whether to escalate instead of answering.
- **The evals runner takes the router as a parameter**, measuring whatever escalation logic the caller
  actually ships, without importing an agent.
- **Seams are injectable for testing** — the LLM, vector store, loader, and every provider factory can
  be replaced with a fake, which is why the suite runs fully offline.

---

## The boundary

`rag_core` must never import a project package. Two mechanisms enforce it:

- `tests/test_boundary.py` parses every module's AST and fails if one imports a known project package
  (AST rather than grep, so prose mentioning a project doesn't trip it).
- CI installs `rag-core` **alone** into a clean environment and runs this suite there, so a stray import
  can't be masked by the workspace venv where both packages are always present.

If the engine seems to need something project-specific, invert the dependency: have the project supply
its own source (`sources.base.REGISTRY.register(...)`), prompt
(`generation.AnswerGenerator(system_prompt=...)`), or router (passed into the evals runner).

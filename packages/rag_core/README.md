# rag-core

**A corpus-agnostic retrieval-augmented-generation engine.** Point it at a corpus with a `config.yml`
and one adapter, and it handles the rest: chunking, embedding, indexing, retrieval, confidence scoring,
grounded answer generation, and evaluation.

It knows nothing about AWS, Jira, LangGraph, or Streamlit. That is enforced, not merely intended — see
[The boundary](#the-boundary) below.

---

## Install

Inside this workspace it's already installed by `uv sync` at the repo root. On its own:

```bash
pip install ./packages/rag_core
```

Installing it alone pulls no agent framework, no web UI and no Jira client — that independence is the
whole reason the package exists. (On Windows also install `pyreadline3`: the Pinecone SDK imports the
POSIX-only `readline` module.)

**Requires:** Python 3.13+, an OpenAI API key and a Pinecone API key (both read from the environment).

---

## The two things a new corpus provides

Everything else is already written.

### 1. A `config.yml`

Non-secret, corpus-shaped settings. Secrets are never read from this file.

```yaml
project: my-corpus
index:
  name: my-corpus-idx
  metric: cosine
  region: us-east-1
models:
  embedding: text-embedding-3-small   # must match between ingest and query
  chat: gpt-4o-mini
chunking:
  size_tokens: 800
  overlap_tokens: 100
  strip_patterns: ['<a name="[^"]*"></a>']   # regexes removed before splitting
retrieval:
  top_k: 4
  min_top_score: 0.35                 # below this, the caller should escalate
sources:
  - id: handbook
    loader: my_loader                 # names an adapter in YOUR registry
    # ...any other keys your adapter needs; they arrive in spec.options
```

### 2. A `DocSource` adapter

A class with a `spec` attribute and a `fetch()` yielding `LoadedDoc`s. This is the only code that knows
how your corpus is obtained — HTTP, git, S3, a local folder:

```python
from rag_core.sources import LoadedDoc

class MySource:
    def __init__(self, spec):
        self.spec = spec

    def fetch(self):
        for name, text in my_documents():
            yield LoadedDoc(
                source_id=self.spec.id,
                source_file=name,
                text=text,
                url=f"https://example.com/{name}",   # used for citations
            )

LOADERS = {"my_loader": MySource}   # maps config.yml's `loader:` to the class
```

Then ingest and query:

```python
from rag_core.config import load_config
from rag_core.sources import build_sources
from rag_core.pipeline import run_ingest

cfg = load_config("path/to/config.yml")
report = run_ingest(cfg, build_sources(cfg.sources, LOADERS))
print(report.summary())
```

```bash
uv run rag-ask "your question" --config path/to/config.yml
```

---

## Commands

| Command | What it does |
|---|---|
| `uv run rag-ask "…" --config <path>` | Retrieve → score confidence → generate a cited answer. Add `-k N` to override `top_k`. |
| `uv run pytest packages/rag_core/tests` | The engine's own suite (101 tests, fully offline). |

`rag-ask` is the engine's only console script; ingestion is driven by the project that owns the corpus,
since only it knows how to fetch the documents.

---

## Layout

```text
src/rag_core/
├── config.py          # RagConfig + nested Chunking/Retrieval configs; YAML + env precedence
├── sources.py         # LoadedDoc, the DocSource protocol, build_sources() — the extension seam
├── pipeline.py        # run_ingest(): source → chunk → embed → upsert, returns an IngestReport
├── chunking/          # markdown header split + token-based sizing, config-driven
├── vectorstore/       # Pinecone create/verify/upsert; dimension guard; create-vs-query split
├── retrieval/         # retriever.py (RetrievedChunk), confidence.py (explainable heuristic)
├── generation/        # answer.py, prompts.py (neutral default), ask.py (the rag-ask CLI)
├── evals/             # EvalCase + hit@k and escalation-accuracy runner, markdown table
└── observability.py   # log_event(): one JSON object per line, CloudWatch-parsable
```

---

## Design notes

- **Configuration precedence is env var > `config.yml` > built-in default.** Secrets come only from the
  environment; the YAML is committed so each corpus is reproducible. Missing required variables are
  collected and reported in a single error, and a `config.yml` path that was given but doesn't exist (or
  isn't a mapping) is a hard failure — silently falling back to defaults would build the wrong index.
- **`RetrievedChunk` is the boundary type.** Nothing downstream of retrieval sees a LangChain `Document`,
  so callers aren't coupled to the vector store's types.
- **The index is created at ingest time, never at query time.** A missing index during a query means
  misconfiguration; auto-creating one would silently return zero results forever.
- **Corpus quirks are configuration, not code.** The `strip_patterns` list exists so a corpus's cleanup
  rules (like awsdocs' `<a name>` heading anchors) live in YAML rather than in the engine.
- **Thresholds are per-project.** Cosine similarity isn't a probability — a usable `min_top_score`
  depends on both the embedding model and the corpus, so it's config, not a constant.
- **Prompts are overridable.** `AnswerPrompts` ships a corpus-neutral default; pass your own to
  `generate_answer` for a domain-specific voice.
- **The evals runner takes the router as a parameter.** It measures whatever escalation logic the caller
  actually ships, without importing an agent.
- **Seams are injectable for testing** — the Pinecone client, the LLM, the vector store and the upsert
  function can all be replaced, which is why the suite runs fully offline.

---

## The boundary

`rag_core` must never import a project package. Two mechanisms enforce it:

- `tests/test_boundary.py` parses every module's AST and fails if one imports a known project package
  (AST rather than grep, so prose mentioning a project doesn't trip it).
- CI installs `rag-core` **alone** into a clean environment and runs this suite there, so a stray import
  can't be masked by the workspace venv where both packages are always present.

If the engine seems to need something project-specific, invert the dependency: have the project pass it
in, as `DocSource`, `AnswerPrompts` and the evals router all do.

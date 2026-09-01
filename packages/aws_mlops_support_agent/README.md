# aws-mlops-support-agent

**The AWS CI/CD support agent: an agentic RAG assistant over the CodeBuild and CodePipeline docs, with
Jira escalation when it can't answer.**

This package is a *consumer* of [`rag-core`](../rag_core/README.md). Everything generic — chunking,
embedding, retrieval, answer generation, evals — lives there. What lives here is everything that is
specifically about AWS docs, this agent's control flow, and Jira.

For the project overview, architecture diagram and deployment guide, see the
[root README](../../README.md).

---

## What this package owns

| Concern | Where | Why it's here and not in the engine |
|---|---|---|
| The `awsdocs_git` source type | `sources/fetch.py` | The archived-repo trick and the docs-URL mapping are specific to `awsdocs`. |
| Corpus + index settings | `config.yml` | Per-project by definition. |
| Jira credentials + `DRY_RUN` | `settings.py` | The engine has no notion of ticketing. |
| The agent's control flow | `agent/` | LangGraph state machine, retries, human-in-the-loop pauses. |
| Ticket drafting + Jira REST | `agent/ticket.py`, `agent/jira_tool.py` | Escalation is this product's feature. |
| The AWS answer prompt | `agent/prompts.py` | Overrides the engine's corpus-neutral default. |
| The eval questions | `evals/dataset.py` | The 15 questions are about *this* corpus. |
| Chat UI | `demo/` | Streamlit, forced into dry-run. |

---

## Commands

| Command | What it does |
|---|---|
| `uv run aws-agent-demo` | Streamlit chat UI on :8501, including the "resolved / open a ticket?" prompt. |
| `uv run aws-agent` | The same graph in the terminal. |
| `uv run aws-agent-ingest` | `RagCore.sync()`: clone → recover → chunk → embed → upsert. Idempotent (deterministic chunk IDs). |
| `uv run aws-agent-evals` | Run the eval set against the live index; writes `evals/results.md`. |
| `uv run pytest packages/aws_mlops_support_agent/tests` | This package's suite (offline — see [Testing](#testing)). |

Ingestion runs separately from serving — the deployed container only serves; the corpus already lives in
Pinecone.

---

## Layout

```text
src/aws_mlops_support_agent/
├── config.yml         # embeddings, llm, vectorstore, splitter, retriever + the two awsdocs sources
├── settings.py        # AgentConfig: Jira vars + DRY_RUN wrapped around RagConfig (cfg.rag)
├── ingest.py          # one call: RagCore.sync() — the source does the rest
├── app.py             # CLI entrypoint: the interrupt/resume loop
├── sources/
│   ├── __init__.py    # importing this registers `awsdocs_git` with rag_core
│   └── fetch.py       # AwsDocsGitSource — a rag_core Source subclass
├── agent/
│   ├── state.py       # AgentState TypedDict; nodes return partial updates
│   ├── nodes.py       # retrieve / answer / confirm_resolution / escalate
│   ├── graph.py       # graph wiring + the two pure routing functions
│   ├── ticket.py      # deterministic TicketDraft builder (no LLM)
│   ├── jira_tool.py   # thin Jira Cloud REST v3 wrapper
│   └── prompts.py     # the AWS voice, overriding rag_core's neutral default
├── demo/              # Streamlit app + the aws-agent-demo launcher shim
└── evals/             # the 15 questions, the runner wiring, and results.md
```

---

## How the corpus is fetched

The `awsdocs` repositories are archived: the markdown was **deleted from the default branch** but still
exists in git history. `AwsDocsGitSource` recovers it without relying on commit messages:

1. If `doc_source/` exists at `HEAD`, use `HEAD` (makes re-runs idempotent, and supports non-archived repos).
2. Otherwise `git rev-list -1 HEAD -- doc_source` finds the commit that last *touched* the directory —
   the deletion commit — so its parent (`<sha>^`) is the last commit where the docs were present. Check
   that out (detached HEAD).

It is a **`rag_core.sources.Source` subclass**, registered as `type: awsdocs_git`, so `RagCore.sync()`
drives it like any built-in source — there is no separate fetch step. `list_files()` does the recovery
above and strips AWS's `<a name="..."></a>` heading anchors in place; `metadata_for()` returns each
file's canonical docs URL, which rag_core merges into **every chunk's `metadata["url"]`**. That is what
lets `agent/ticket.py` cite a real docs link straight off a retrieved chunk.

Adding a third AWS guide is one entry in `AWSDOCS_REPOS` (`sources/fetch.py`) plus one `sources:` entry
in `config.yml`:

```yaml
sources:
  - type: awsdocs_git
    id: codebuild          # keys into AWSDOCS_REPOS
    path: data/aws_docs    # clones live at <path>/<id>/
```

---

## Configuration

Two layers, with **env var > `config.yml` > default** precedence:

- `config.yml` (committed, ships inside the wheel) — `embeddings`, `llm`, `vectorstore`, `splitter`,
  `retriever`, `sources`. This is rag_core's generic schema; see
  [`config.example.yaml`](../rag_core/config.example.yaml).
- Environment — secrets only (`OPENAI_API_KEY`, `PINECONE_API_KEY`, the four `JIRA_*` vars), plus optional
  overrides of any non-secret key.

`AgentConfig` **wraps** `RagConfig` rather than extending it: `cfg.rag` is what gets handed to every
engine function, while `cfg.jira_*` and `cfg.dry_run` stay on this side of the boundary.

**`DRY_RUN` fails safe.** Only an explicit `false` / `0` / `no` disables it — unset, empty, or a typo like
`flase` all keep dry-run *on*, so a config mistake can never silently create real tickets. The Streamlit
demo goes further and forces `dry_run=True` regardless of the environment.

Verify what's actually loaded (secrets masked):

```bash
uv run python -m aws_mlops_support_agent.settings
```

---

## The agent graph

`START → retrieve →(confident?)→ answer → confirm_resolution →(resolved / ticket / retry)→ END | escalate`

- **Escalation has exactly three triggers**, each decided in one pure routing function: low retrieval
  confidence, the user asking for a ticket, or retries exhausted (`MAX_ATTEMPTS = 2`).
- **Retries widen the net** rather than repeating an identical search: `k = top_k + 2 × attempts`.
- **The pause is a real interrupt.** `interrupt()` checkpoints the state and returns control to the
  caller; resuming with `Command(resume=...)` re-runs the node from its start — which is why interrupt
  nodes do nothing expensive beforehand.
- **No real ticket is created without confirmation.** In live mode `escalate` raises a second interrupt
  showing the draft before anything is POSTed.

Render the graph (no API keys needed — fakes are injected):

```bash
uv run python -m aws_mlops_support_agent.agent.graph
```

---

## Testing

```bash
uv run pytest packages/aws_mlops_support_agent/tests -q
```

Fully offline by design: `tests/conftest.py` provides a `make_config()` factory, and the graph accepts stub
retriever / answerer / Jira functions, so the whole interrupt-and-resume flow runs with no network and no
keys. `tests/jira_live_check.py` is the deliberate exception — a manual script that creates a **real**
ticket, named so pytest never collects it.

> **Known breakage.** `tests/conftest.py` still builds a `RagConfig` from the pre-migration flat schema
> (`ChunkingConfig`, `RetrievalConfig`, `openai_api_key`, …), all of which rag_core removed. It fails to
> import, which blocks collection of `test_graph.py`, `test_jira_tool.py`, `test_routing.py` and
> `test_ticket.py`. The other files (33 tests) pass with `--noconftest`. Porting the fixtures to the new
> `embeddings` / `llm` / `vectorstore` / `splitter` / `retriever` blocks is outstanding work.

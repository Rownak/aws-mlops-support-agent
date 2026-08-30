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
| awsdocs git-history recovery | `sources/fetch.py` | The archived-repo trick is specific to `awsdocs`. |
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
| `uv run aws-agent-ingest` | Fetch → chunk → embed → upsert the corpus. Idempotent (deterministic chunk IDs). |
| `uv run aws-agent-evals` | Run the eval set against the live index; writes `evals/results.md`. |
| `uv run pytest packages/aws_mlops_support_agent/tests` | This package's suite (56 tests, offline). |

Ingestion runs separately from serving — the deployed container only serves; the corpus already lives in
Pinecone.

---

## Layout

```text
src/aws_mlops_support_agent/
├── config.yml         # index, models, chunking, retrieval, and the two awsdocs sources
├── settings.py        # AgentConfig: Jira vars + DRY_RUN wrapped around RagConfig (cfg.rag)
├── ingest.py          # wires this project's sources into rag_core's pipeline
├── app.py             # CLI entrypoint: the interrupt/resume loop
├── sources/
│   └── fetch.py       # AwsDocsGitSource — implements rag_core's DocSource protocol
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

Adding a third AWS guide is a `config.yml` entry, not a code change: `LOADERS` maps the `loader:` name
`awsdocs_git` to this class, and `rag_core.sources.build_sources` does the wiring.

---

## Configuration

Two layers, with **env var > `config.yml` > default** precedence:

- `config.yml` (committed, ships inside the wheel) — index name, models, chunking, retrieval, sources.
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

Fully offline: `tests/conftest.py` provides a `make_config()` factory, and the graph accepts stub
retriever / answerer / Jira functions, so the whole interrupt-and-resume flow runs with no network and no
keys. `tests/jira_live_check.py` is the deliberate exception — a manual script that creates a **real**
ticket, named so pytest never collects it.

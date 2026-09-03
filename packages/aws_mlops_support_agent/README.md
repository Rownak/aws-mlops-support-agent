# aws-mlops-support-agent

**The AWS CI/CD support agent: an agentic RAG assistant over the CodeBuild and CodePipeline docs, with
Jira escalation when it can't answer.**

This package is a *consumer* of [`rag-core`](../rag_core/README.md). Everything generic — chunking,
embedding, retrieval, answer generation, evals — lives there. What lives here is everything that is
specifically about AWS docs, this agent's control flow, and Jira.

Retrieval techniques used here are scored on labelled IR data by
[`rag_bench_eval`](../rag_bench_eval/README.md) before they reach production.

> **Status: 🟡 catching up.** This package has not been re-verified since `rag_core` was extended for the
> benchmark; its test suite currently fails to collect. See [Testing](#testing).

For the monorepo overview and deployment guide, see the [root README](../../README.md).

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

Adding a third AWS guide is one `sources:` entry in `config.yml` — `git_url` and `docs_base_url` are
config, not code:

```yaml
sources:
  - type: awsdocs_git
    id: codebuild
    path: data/aws_docs    # clones live at <path>/<id>/
    git_url: https://github.com/awsdocs/aws-codebuild-user-guide.git
    docs_base_url: https://docs.aws.amazon.com/codebuild/latest/userguide/
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

Copy [`.env.example`](../../.env.example) to `.env` and fill in the two required keys — everything else
has a sensible default.

| Variable | Required? | Default (from `config.yml`) | Description |
|----------|-----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ | — | OpenAI key for chat + embeddings. |
| `PINECONE_API_KEY` | ✅ | — | Pinecone key for the vector index. |
| `AWS_REGION` | | `us-east-1` | AWS region for the serverless index. |
| `DRY_RUN` | | `true` | Safety gate: Jira tickets are only *logged* unless explicitly set to `false`. |

<details>
<summary>Optional: tuning overrides, Jira ticket creation & LangSmith tracing</summary>

Models, the index name and the providers live in `config.yml`. The tuning knobs below can also be set by
environment variable, which is useful for experiments without editing (and committing) the file:

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_TOP_K` | `4` | Chunks retrieved per query. |
| `RAG_MIN_TOP_SCORE` | `0.35` | Confidence threshold below which the agent escalates. |
| `RAG_CHUNK_SIZE` | `800` | Chunk size at ingest time. |
| `RAG_CHUNK_OVERLAP` | `100` | Overlap between adjacent chunks. |

| Variable | Required? | Description |
|----------|-----------|-------------|
| `JIRA_BASE_URL` | Only to create real tickets | e.g. `https://your-site.atlassian.net`. |
| `JIRA_EMAIL` | Only to create real tickets | The email you log into Jira with. |
| `JIRA_API_TOKEN` | Only to create real tickets | API token (not your password) — [create one here](https://id.atlassian.com/manage-profile/security/api-tokens). |
| `JIRA_PROJECT_KEY` | Only to create real tickets | Project key to file under, e.g. `SUP`. |
| `LANGSMITH_TRACING` | Optional | Set to `true` to trace every graph run. |
| `LANGSMITH_API_KEY` | Optional | LangSmith key ([free account](https://smith.langchain.com)). |
| `LANGSMITH_PROJECT` | Optional | Trace bucket name (defaults to `default`). |

`.env.example` has step-by-step setup notes for each of these. Never commit real secrets — `.env` is
gitignored.
</details>

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

## Evaluation results

A 15-question eval set (12 in-corpus with expected doc files, 3 off-corpus negatives) measures whether
retrieval surfaces the right docs (**hit@4**) and whether the agent escalates when it should. Runner:
`uv run aws-agent-evals` (embedding calls only, no LLM). Full table:
[`evals/results.md`](src/aws_mlops_support_agent/evals/results.md).

| Metric | Result | Notes |
|--------|--------|-------|
| **Hit@4** (in-corpus) | **11 / 12** | Correct doc in the top 4 chunks for all but one question. |
| **Escalation accuracy** | **12 / 15** | All 3 misses are off-corpus questions the retriever scored too confidently. |

**Honest caveat:** the 3 escalation failures are off-corpus questions (EKS, SageMaker, account password)
that scored *above* the `0.35` confidence threshold — the current heuristic (top cosine score) doesn't
cleanly separate "irrelevant but adjacent" AWS topics. Tuning that threshold / adding a reranker is a
deliberate next step, not a solved problem.

This eval measures *this corpus, this agent*. The technique-level question — is dense better than BM25,
does reranking help — is answered separately on labelled IR data by
[`rag_bench_eval`](../rag_bench_eval/README.md), whose current finding is that the best technique is
corpus-dependent, so any candidate still has to be re-verified here.

---

## Testing

```bash
uv run pytest packages/aws_mlops_support_agent/tests -q
```

Fully offline by design: `tests/conftest.py` provides a `make_config()` factory, and the graph accepts stub
retriever / answerer / Jira functions, so the whole interrupt-and-resume flow runs with no network and no
keys. `tests/jira_live_check.py` is the deliberate exception — a manual script that creates a **real**
ticket, named so pytest never collects it.

> **⚠️ Known breakage — this package is mid-migration.** `tests/conftest.py` still builds a `RagConfig`
> from the pre-migration flat schema (`ChunkingConfig`, `RetrievalConfig`, `openai_api_key`, …), all of
> which rag_core removed. It fails to import, which blocks collection of the whole package. Some test
> modules also still import `rag_core.retrieval.*`, a path that is now `rag_core.retriever.*`.
>
> Separately, `rag_core` is being extended for [`rag_bench_eval`](../rag_bench_eval/README.md) — a
> `Retriever` protocol, new retrieval implementations, and a rename inside `rerank.py`
> (`BaseReranker` → `RelevanceScorer`, …). Those changes are meant to be additive, and every module in
> this package still imports cleanly against current `rag_core`, but the agent has **not** been
> re-verified end to end since. Treat it as untested on this branch.
>
> Porting the fixtures to the nested `embeddings` / `llm` / `vectorstore` / `splitter` / `retriever`
> blocks, then re-running ingest → query → escalate, is the outstanding work.

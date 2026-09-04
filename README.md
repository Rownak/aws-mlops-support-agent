# RAG Pipeline Monorepo

**A reusable RAG engine, a benchmark that measures its retrieval quality, and a production agent built
on top of it.**

![Python 3.13](https://img.shields.io/badge/python-3.13-blue)
![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-1c3d5a)
![Pinecone](https://img.shields.io/badge/vector%20db-Pinecone-6f42c1)
![BEIR](https://img.shields.io/badge/benchmark-BEIR-critical)
![Deploy](https://img.shields.io/badge/deploy-ECS%20Fargate-ff9900)
![Code license](https://img.shields.io/badge/code-MIT-green)

---

## What it does & why

Three packages, one dependency direction:

| Package | What it is |
|---|---|
| **[`rag_core`](packages/rag_core/README.md)** | A corpus-agnostic RAG engine. Hand it a `config.yaml` and it loads, chunks, embeds, indexes, retrieves, reranks, scores confidence, and generates cited answers. Knows nothing about AWS, Jira, LangGraph, or BEIR. |
| **[`rag_bench_eval`](packages/rag_bench_eval/README.md)** | A retrieval benchmark over labelled IR datasets (BEIR/NFCorpus, CQADupStack). Scores each technique — BM25, dense, RRF hybrid, reranking — with nDCG@10 against human relevance judgements. |
| **[`aws_mlops_support_agent`](packages/aws_mlops_support_agent/README.md)** | The production consumer: an agentic assistant that answers AWS CI/CD questions from the official docs with citations, and files a Jira ticket for a human when it can't. |

The point of the split is that **retrieval decisions get measured before they ship.** A change to
`rag_core`'s retrieval is scored on labelled data by `rag_bench_eval` and only then carried into the
agent. Both apps consume the same engine; neither can be imported by it.

It's a portfolio project showcasing end-to-end **RAG**, **retrieval evaluation**, **agentic workflows
(LangGraph state machines)**, and **MLOps on AWS**.

**Key features**

- 🧩 **Reusable engine** — `pip install ./packages/rag_core` and point it at any corpus.
  Provider-switchable (OpenAI / Ollama / Google / HuggingFace), hash-based incremental ingest, hybrid
  retrieval, reranking, confidence scoring. Two independent apps prove the seam.
- 📏 **Measured retrieval, not eyeballed** — every technique scored on the same queries against the same
  labels, with the BM25 baseline validated against a published figure first.
- 🔎 **Grounded answers with citations** — every answer links back to the exact AWS doc section it used.
- 🧠 **Agentic escalation** — a LangGraph state machine loops, retries, and escalates when confidence is
  low, retries run out, or you ask it to.
- 🎫 **Jira ticket drafting** — turns an unresolved question into a structured ticket (dry-run by
  default, so no ticket is created unless you explicitly opt in).
- 🚀 **Real deployment path** — containerized, pushed to ECR via GitHub Actions, running on ECS Fargate
  with secrets in AWS Secrets Manager.

---

## Architecture at a glance

```text
                 ┌──────────────────────────────────────┐
                 │              rag_core                │
                 │  sources → loaders → processing      │
                 │  embeddings → vectorstores           │
                 │  retriever (protocol + techniques)   │
                 │  generation · evals · config         │
                 └──────────────────────────────────────┘
                       ▲                        ▲
        measures ──────┘                        └────── consumes
                       │                        │
    ┌──────────────────────────┐   ┌──────────────────────────────┐
    │      rag_bench_eval      │   │   aws_mlops_support_agent    │
    │  BEIR datasets, qrels    │   │  AWS docs source, LangGraph  │
    │  pipelines, nDCG@10      │   │  Jira escalation, Streamlit  │
    └──────────────────────────┘   └──────────────────────────────┘
```

Retrieval techniques implement one protocol —
`search(query, k) -> list[SearchResult]` — so they **compose by nesting**: a reranker wraps an RRF
fusion which wraps BM25 and dense. That is what lets the benchmark declare a whole pipeline in YAML and
the agent adopt whichever shape won.

The agent itself is a small **state machine**: retrieve → answer → confirm → (maybe) escalate. Solid
arrows are always-taken; dashed arrows are **conditional**.

```mermaid
graph TD;
    __start__([start]):::first
    retrieve(retrieve)
    answer(answer)
    confirm_resolution(confirm_resolution)
    escalate(escalate)
    __end__([end]):::last
    __start__ --> retrieve;
    retrieve -.->|confident| answer;
    retrieve -.->|low confidence| escalate;
    answer --> confirm_resolution;
    confirm_resolution -.->|resolved| __end__;
    confirm_resolution -.->|retry, attempts left| retrieve;
    confirm_resolution -.->|open a ticket / attempts exhausted| escalate;
    escalate --> __end__;
    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0
    classDef last fill:#bfb6fc
```

Full walkthrough of the nodes, escalation triggers and interrupt/resume flow:
[`aws_mlops_support_agent/README.md`](packages/aws_mlops_support_agent/README.md).

**Tech stack:** Python 3.13 · [LangGraph](https://langchain-ai.github.io/langgraph/) (orchestration) ·
[LangChain](https://python.langchain.com/) (RAG plumbing) · [Pinecone](https://www.pinecone.io/)
serverless (vector DB, swappable for Pinecone Local) · OpenAI via `langchain_openai` (chat + embeddings;
Ollama / Google / HuggingFace selectable from config) · `rank_bm25` + numpy + sentence-transformers
(benchmark retrieval) · Jira Cloud REST API · Docker · ECS Fargate · GitHub Actions + ECR ·
CloudWatch + [LangSmith](https://smith.langchain.com/) (logging & tracing).

---

## Quick start

**Prerequisites:** Python 3.13, [`uv`](https://docs.astral.sh/uv/). For the agent: an **OpenAI API key**
and a **Pinecone API key** (free serverless tier is enough); Jira and LangSmith are optional. For the
benchmark: a local [Ollama](https://ollama.com/) for the dense pipelines — no cloud keys needed.

```bash
git clone https://github.com/Rownak/aws-mlops-support-agent.git
cd aws-mlops-support-agent
uv sync                 # install the whole workspace from the lockfile
```

**Run the agent** *(⚠️ mid-migration on this branch — see [Roadmap / status](#roadmap--status))*:

```bash
cp .env.example .env    # then fill in OPENAI_API_KEY and PINECONE_API_KEY

uv run aws-agent-ingest # fetch AWS docs, chunk, embed, upsert to Pinecone (one-time)
uv run aws-agent-demo   # open the chat demo  (or: uv run aws-agent, for the terminal)
```

Within a few minutes you have a running chat UI backed by a real vector index. The ingest step pulls the
AWS docs at build time (never committed — see [License & attribution](#license--attribution)) and is safe
to re-run (idempotent upserts).

**Run the benchmark:**

```bash
ollama pull nomic-embed-text

uv run rag-bench-eval download                            # fetch + cache NFCorpus
uv run rag-bench-eval run --experiment bm25 --limit 20    # smoke test
uv run rag-bench-eval run --all                           # full sweep, all pipelines
uv run rag-bench-eval report                              # regenerate results/results.md
```

**Query any corpus through the engine alone:**

```bash
uv run rag-ask "What phases can I define in a buildspec file?" \
  --config packages/aws_mlops_support_agent/src/aws_mlops_support_agent/config.yml
```

---

## `rag_core` — the engine

A complete RAG pipeline in its own right: `pip install ./packages/rag_core`, hand it a `config.yaml`, and
it loads, chunks, embeds, indexes, retrieves, reranks and answers.

```python
from rag_core import RagCore

rag = RagCore("config.yaml")
rag.sync()                                  # reconcile the index against configured sources
print(rag.query("your question").formatted())
```

Highlights:

- **One YAML file configures everything**, with **env var > `config.yaml` > default** precedence per
  field. Secrets are read *only* from the environment and missing ones fail fast, all reported in one
  error. Switching between local dev (Ollama + Pinecone Local, no keys) and a managed deployment needs
  no code change.
- **Content hash is the unit of ingest identity** — re-runs are free for unchanged files, a changed file
  replaces its older version, and a run that died mid-write is rolled back rather than left truncated.
- **A `Retriever` protocol with composable implementations** — BM25, dense, RRF fusion, reranking — plus
  the production path (`retrieve` / `retrieve_scored` over a live Pinecone index, with MMR, metadata
  filters and normalized scores).
- **Every step is standalone.** `RagCore` is a convenience facade; `retriever.retrieve`,
  `assess_confidence` and `AnswerGenerator.generate` are each independently callable, which is what lets
  the agent interleave its own control flow and escalate on a weak match instead of answering.
- **`evals/`** carries both IR metrics (nDCG, recall, MRR, precision + a generic `ir_runner`) and the
  hit@k / escalation-accuracy runner, which takes the router as a parameter so it measures whatever
  logic the caller actually ships.

Details, config reference and design notes: [`packages/rag_core/README.md`](packages/rag_core/README.md).

---

## `rag_bench_eval` — measuring retrieval

Retrieval quality sets the ceiling on any RAG system: if the right document never reaches the context
window, no amount of prompting recovers it. Yet retrieval changes are usually judged by eyeballing a
handful of queries, which cannot separate a real gain from noise.

This package replaces that with measurement. Each technique is a named pipeline in `benchmark.yaml`, run
over every query in a labelled dataset and scored against human relevance judgements — nDCG@10 as the
primary metric, alongside recall@100, MRR@10 and precision@10, plus latency and LLM-call cost. The BM25
baseline is validated against a published figure (NFCorpus ≈ 0.32) first, which is what makes the rest of
the numbers trustworthy.

**Results so far** — NFCorpus (323 queries) and CQADupStack Programmers (876 queries), nDCG@10:

| pipeline | NFCorpus | CQADupStack |
|---|---|---|
| cross-encoder rerank over RRF hybrid | **0.3554** | 0.3932 |
| RRF hybrid (BM25 + dense) | 0.3413 | 0.3651 |
| dense | 0.3408 | **0.4279** |
| bi-encoder rerank over dense | 0.3408 | 0.4279 |
| BM25 | 0.3052 | 0.2680 |

Two findings worth the whole exercise: **dense beats BM25 on both corpora** — the only result that held
its direction — but **the winner does not transfer.** Cross-encoder reranking tops NFCorpus and *loses*
to plain dense on CQADupStack, at roughly 8-12× the latency. Reranking also costs recall@100 every time,
since it reorders a truncated candidate pool and cannot recover what the first stage missed.

Full tables, findings and the technique-by-technique explanation:
[`packages/rag_bench_eval/README.md`](packages/rag_bench_eval/README.md).

---

## `aws_mlops_support_agent` — the production consumer

Ask it *"How do I cache dependencies between CodeBuild builds?"* and it searches the official AWS
CodeBuild and CodePipeline documentation, writes a grounded answer **with citations**, and asks whether
that resolved your issue. If it didn't — or the docs clearly don't cover the question — it drafts a
**Jira support ticket** (problem, docs already checked, suggested next steps) and hands off to a human.

Everything AWS-, Jira- or LangGraph-specific lives in this package: the `awsdocs_git` source that
recovers markdown deleted from archived repos, the agent graph and its three escalation triggers, ticket
drafting, the AWS answer prompt, the eval questions, and the Streamlit UI.

> **Live demo note:** the hosted demo forces Jira into **dry-run mode**, so visitors can't create real
> tickets — the drafted payload is logged instead.

**Corpus eval** — a 15-question set (12 in-corpus, 3 off-corpus negatives) measuring whether retrieval
surfaces the right docs and whether the agent escalates when it should:

| Metric | Result | Notes |
|--------|--------|-------|
| **Hit@4** (in-corpus) | **11 / 12** | Correct doc in the top 4 chunks for all but one question. |
| **Escalation accuracy** | **12 / 15** | All 3 misses are off-corpus questions the retriever scored too confidently. |

**Honest caveat:** those 3 failures are off-corpus questions (EKS, SageMaker, account password) scoring
*above* the `0.35` threshold — the current heuristic (top cosine score) doesn't cleanly separate
"irrelevant but adjacent" AWS topics. Fixing that is what `rag_bench_eval` exists to inform.

Configuration, commands, the graph walkthrough and corpus-fetch mechanics:
[`packages/aws_mlops_support_agent/README.md`](packages/aws_mlops_support_agent/README.md).

---

## Project structure

A **uv workspace** of three independently installable packages.

```text
packages/
├── rag_core/                     # the generic RAG engine — installable on its own
│   └── src/rag_core/
│       ├── config/               # typed config tree: YAML + ${VAR} + env precedence, fail-fast secrets
│       ├── pipeline.py           # RagCore facade: ingest_documents / ingest_directory / sync / query
│       ├── sources/              # Source ABC + lazy registry — the extension seam for a new corpus
│       ├── loaders/ processing/  # bytes → markdown; content hashing + chunking
│       ├── embeddings/ llm/      # provider factories (openai, ollama, google, huggingface, …)
│       ├── vectorstores/         # Pinecone index lifecycle + hash-based ingest-state tracking
│       ├── retriever/            # base protocol, bm25, dense, fusion, rerank, factory
│       │                         #   + retrieve/hybrid/confidence (the live-index production path)
│       ├── generation/           # cited Answer objects, context budget, the `rag-ask` CLI
│       └── evals/                # metrics + ir_runner · hit@k + escalation-accuracy runner
├── rag_bench_eval/               # retrieval benchmark on labelled IR datasets
│   ├── benchmark.yaml            # resources, pipelines, sweep list, metrics
│   ├── results/                  # runs/*.json + comparison tables
│   └── src/rag_bench_eval/       # datasets, config, resources, caches, evaluator, report, CLI
└── aws_mlops_support_agent/      # AWS docs corpus + support agent
    └── src/aws_mlops_support_agent/
        ├── config.yml            # embeddings, llm, vectorstore, splitter, retriever + awsdocs sources
        ├── settings.py           # AgentConfig: Jira vars + DRY_RUN wrapped around RagConfig (cfg.rag)
        ├── sources/              # AwsDocsGitSource — registers `awsdocs_git` with rag_core
        ├── agent/                # LangGraph state machine, Jira tool, ticket builder, AWS prompt
        ├── demo/ evals/          # Streamlit chat UI (dry-run forced) · this corpus's 15 questions
        └── ingest.py, app.py     # `RagCore.sync()` · CLI entrypoint
deploy/                           # ECR + ECS Fargate runbooks and task definition
```

**The rule that keeps it honest:** dependencies point one way only — app → `rag_core`, never the reverse.
It's enforced structurally, not by convention: a test parses every `rag_core` module and fails the build
if one imports an app package, and CI additionally installs `rag-core` *alone* and runs its suite with no
app package present.

A reusable technique or metric belongs in `rag_core`; a dataset or experiment workflow belongs in
`rag_bench_eval`. That is why the IR metrics and the `Retriever` implementations live in the engine while
BEIR loading and `benchmark.yaml` do not.

---

## Adding a new RAG project

Because the engine is corpus-agnostic, most new corpora are **config only** — the built-in `local` and
`s3` sources cover files on disk or in a bucket:

```python
from rag_core import RagCore

rag = RagCore("config.yaml")
rag.sync()
print(rag.query("your question").formatted())
```

A corpus obtained some other way (this project's archived-git-history trick, an API, a crawler) needs
**one adapter**: subclass `rag_core.sources.Source`, implement `list_files()` (and optionally
`metadata_for()` to attach a citable URL to every chunk), and register it under a `type:` name — see
[`sources/fetch.py`](packages/aws_mlops_support_agent/src/aws_mlops_support_agent/sources/fetch.py).
`RagCore.sync()` then drives it like any built-in source; ingest is content-hash based, so re-runs are
free for unchanged files.

Optionally pass a domain-specific `system_prompt` to `AnswerGenerator`; the default is corpus-neutral.
Anything beyond that — an agent, a UI, ticketing — is yours to add, exactly as the AWS agent does.

---

## Testing

```bash
uv run pytest            # fully offline (fakes injected for retriever, LLM, Pinecone, Jira)
uv run ruff check .      # lint
uv run ruff format .     # format
```

Tests favor small, injectable fakes over mocking the network — the graph accepts stub
retriever/answerer/Jira functions, so the full interrupt-and-resume flow is tested with no API keys.
Metrics are the deliberate exception to "prefer eval scripts over unit tests": they get real tests
against hand-computed rankings, because a wrong nDCG would invalidate every number in the benchmark.

The suites live with their packages (`packages/*/tests/`); one `uv run pytest` from the root runs all
three.

> **Known gap:** the agent package's `tests/conftest.py` imports `ChunkingConfig` / `RetrievalConfig`,
> which the config migration removed, so `uv run pytest` currently fails at collection for that package —
> a root-level run stops there rather than reporting the other two suites. Run them directly
> (`uv run pytest packages/rag_core/tests packages/rag_bench_eval/tests`) until the fixtures are ported
> to the nested `embeddings` / `llm` / `vectorstore` / `splitter` / `retriever` blocks. Tracked in
> [Roadmap / status](#roadmap--status).

To reproduce CI's independence check — `rag_core`'s tests passing with **only** the engine installed:

```bash
uv venv --python 3.13 /tmp/isolated
VIRTUAL_ENV=/tmp/isolated uv pip install ./packages/rag_core pytest
VIRTUAL_ENV=/tmp/isolated uv run --no-project python -m pytest packages/rag_core/tests -q
```

On Windows add `pyreadline3` to that install — the Pinecone SDK imports the POSIX-only `readline` module.

---

## Deployment

The agent ships as a container and runs on **ECS Fargate**, with images built and pushed to **ECR** by
**GitHub Actions** (OIDC — no AWS keys stored in GitHub). Secrets come from **AWS Secrets Manager**.

```bash
docker build -t aws-mlops-support-agent .          # multi-stage build, non-root, serves Streamlit on 8501
docker run --env-file .env -p 8501:8501 aws-mlops-support-agent
```

The image builds the whole workspace and installs the packages non-editable, so the runtime stage carries
only the venv — including each package's `config.yml`. The container runs the `aws-agent-demo` console
script, so it never hardcodes a source path.

Full, step-by-step AWS runbooks:

- [`deploy/aws_ecr_setup.md`](deploy/aws_ecr_setup.md) — one-time ECR + GitHub OIDC setup.
- [`deploy/aws_ecs_setup.md`](deploy/aws_ecs_setup.md) — Secrets Manager, task definition, service,
  verification, and cost control (scale-to-zero).

Ingestion runs **separately** from serving (locally or as a scheduled job) — the container only serves;
the corpus already lives in Pinecone.

---

## Roadmap / status

| Package | Status |
|---|---|
| `rag_core` | 🟢 **Stable.** Standalone, provider-switchable engine; its own suite passes. |
| `rag_bench_eval` | 🟢 **Active.** Five retrieval pipelines swept across two BEIR datasets. |
| `aws_mlops_support_agent` | 🟢 **Verified against current `rag_core`.** Test fixtures ported; ingest, query and escalate all re-run end to end. |


Backlog / next steps:

- **Fix 4 files failing ingest with `UnicodeDecodeError`.** `markitdown`'s `PlainTextConverter`
  (via `charset_normalizer`/Magika charset detection) misidentifies the encoding of 4 of 182
  CodeBuild doc files on this Windows environment — the files are valid UTF-8, but conversion
  decodes with the wrong charset and fails. Pre-existing, unrelated to the `rag_core` migration;
  found while re-verifying ingest end to end.
- Carry the benchmark's winning retrieval shape into the agent's config and verify it on the AWS corpus —
  where the two datasets already disagree, the AWS numbers decide.
- HyDE and multi-query pipelines (query expansion via LLM), with per-run LLM-call accounting.
- Tune the confidence threshold / add reranking to fix off-corpus escalation (see caveat above).
- Expand the AWS corpus (Step Functions, Lambda, EventBridge, S3, CloudFormation, IAM).
- Swap Pinecone → OpenSearch Serverless / pgvector for an all-AWS stack.
- Schedule ingestion via EventBridge; response streaming in the UI.

---

## FAQ / troubleshooting

<details>
<summary>Ingestion fails or the index looks empty</summary>

`aws-agent-ingest` clones the AWS doc repos with full history and checks out the commit *before* they were
archived (the content was later stripped from the default branch). If it can't find `doc_source/*.md`,
make sure `git` is on your PATH and the clone completed. Re-running is safe — upserts are idempotent.
</details>

<details>
<summary>Answers come back with no source labels</summary>

Chunk metadata keys changed across refactors (`service` → `source_id` → rag_core's `source` / `url` /
`file_hash`). An index built by an older version won't match. Re-run `uv run aws-agent-ingest` to
re-embed the corpus under the current keys.
</details>

<details>
<summary>The demo answers but never offers a ticket</summary>

That's expected when retrieval is confident and you mark the issue resolved. To see escalation, ask an
off-topic question (low confidence), or choose "open a ticket" at the confirm prompt. In the demo,
tickets are always dry-run (logged, not created).
</details>

<details>
<summary>A benchmark <code>dense</code> run fails or hangs</summary>

Dense pipelines embed the whole corpus through Ollama — it must be running locally with
`nomic-embed-text` pulled. The first cross-encoder run additionally downloads its model (~90MB). Corpus
vectors are cached to disk after the first run, keyed by model plus a corpus hash, so re-runs are fast
and an edited corpus invalidates the cache rather than silently reusing stale vectors.
</details>

<details>
<summary><code>docker --env-file</code> rejects my <code>.env</code></summary>

Docker is stricter than python-dotenv about whitespace in variable *names* (e.g. `LANGSMITH_TRACING =`
with a space before `=`). Remove stray spaces around `=` in `.env`.
</details>

<details>
<summary>Answers seem outdated</summary>

The AWS docs corpus is frozen at roughly 2023 (the last content before the source repos were archived).
Fine for a demo; not a substitute for current AWS documentation.
</details>

---

## License & attribution

- **Code:** [MIT](LICENSE) — free to use, modify, and learn from.
- **Use of AI for development:** the architecture and task plan were designed by the developer, drafted
  with Claude, then reviewed and edited by the developer. Tasks are executed with Claude Code in plan
  mode — the developer reviews and edits each plan before execution.
- **AWS documentation content:** the ingested docs (AWS CodeBuild & CodePipeline user guides) are
  © Amazon Web Services, licensed under
  **[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)**. This repository **does not
  redistribute** the raw doc text — it is fetched at build time by `aws-agent-ingest` from the public
  `awsdocs` GitHub repositories (https://github.com/awsdocs). Attribution: *"AWS Documentation,"*
  © Amazon Web Services, Inc., used under CC BY-SA 4.0.
- **Benchmark data:** BEIR/NFCorpus and CQADupStack are CC BY-SA and likewise fetched at run time, never
  committed.

Built as a demo project — RAG, retrieval evaluation, agentic AI, and MLOps on AWS.

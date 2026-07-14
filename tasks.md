# Tasks — AWS Support Agent

Work through these **one at a time**. For each task: Claude Code proposes a plan →
I approve → it writes simple, explained code → I review, test, adjust → I commit.
Mark `[x]` when committed. This is a living list — refine later tasks as I learn.

Each task is sized to ONE concept I can hold in my head and test in isolation.

**Quick-build rule:** anything not needed to (a) demo the core flow or (b) teach me
a concept I set out to learn gets pushed to the backlog. Ship the happy path first.

---

## Phase 0 — Project setup
- [x] **0.1** Initialize repo scaffolding with `uv`: run `uv init`, create the
  folder structure (`src/ingest`, `src/agent`, `src/tools`, `src/evals`, `tests`),
  add dependencies via `uv add` (managed in `pyproject.toml` / `uv.lock`),
  `.gitignore` (include `.env`, `data/`, `__pycache__`, `.venv`), `.env.example`
  listing the env vars I'll need (no values), and a minimal README stub.
  No app logic yet.
- [x] **0.2** Set up config loading: a small module that reads env vars (OpenAI
  API key, chat + embedding model names, Pinecone key/index name, AWS region,
  Jira creds, `DRY_RUN` default true) and fails clearly if a required one is
  missing. Add `ruff` config (as a dev dependency via `uv add --dev ruff`).

## Phase 1 — Ingestion (RAG corpus)
- [x] **1.1** Doc fetch step: clone `aws-codebuild-user-guide` and
  `aws-codepipeline-user-guide` with full history, detect the archival
  ("delete content directory") commit, check out the commit before it, and expose
  the local paths to `doc_source/*.md`. Repo list configurable. Clone into
  gitignored `data/` (CC BY-SA 4.0 — pull at build time, don't commit raw docs).
- [x] **1.2** Markdown parsing + structure-aware chunking: load `doc_source/*.md`,
  split with `MarkdownHeaderTextSplitter` then a secondary character splitter for
  oversized sections. Target ~500–1000 tokens, ~10–15% overlap.
- [x] **1.3** Metadata: attach to each chunk the service (repo), source filename,
  section heading, and the reconstructed `docs.aws.amazon.com` URL. (Needed later
  for citations and the Jira "docs checked" field.)
- [x] **1.4** Embeddings + Pinecone: create the Pinecone serverless index with the
  correct dimension for the chosen OpenAI embedding model (e.g.
  `text-embedding-3-small` → 1536), embed chunks via `langchain_openai`, upsert
  with metadata. Make it safe to re-run (idempotent upserts / deterministic IDs).
  Record the embedding model name in config so ingestion and query always match.
- [x] **1.5** Ingestion sanity check: a tiny script/CLI that runs a sample query
  against Pinecone and prints the top-k chunks + metadata, so I can eyeball
  retrieval quality before building the agent.

## Phase 2 — RAG retrieval core
- [x] **2.1** Retriever function: given a question, embed it (same OpenAI
  embedding model as ingestion) and return top-k chunks with scores + metadata.
  Standalone and unit-testable.
- [x] **2.2** Answer generation: prompt the OpenAI chat model (via
  `langchain_openai`) with the question + retrieved chunks; return an answer that
  cites which doc/section it used. Keep the prompt in a readable, editable place.
- [x] **2.3** Confidence signal: a simple, explainable measure of retrieval
  confidence (e.g., top score threshold and/or score gap) to later drive
  escalation. Start simple; document the heuristic.

## Phase 3 — Agent state machine (LangGraph)
- [x] **3.1** Define the graph state schema (question, retrieved docs, answer,
  attempt count, confidence, resolved flag, user satisfaction, ticket draft).
  Explain each field.
- [x] **3.2** Implement the `retrieve` node (wraps 2.1) and `answer` node
  (wraps 2.2). Wire a minimal linear graph and run it end to end.
- [x] **3.3** Add the `confirm_resolution` node with a **user-input step**
  (LangGraph `interrupt` / human-in-the-loop): after an answer, ask whether the
  issue is resolved or the user wants a ticket. Explain how interrupts and
  checkpointing work — this is the core "agentic" learning piece.
- [x] **3.4** Conditional edges + escalation routing: loop retrieve→answer up to
  N attempts; route to `escalate` when attempts are exhausted, OR retrieval
  confidence is low, OR the user asks for a ticket. Explain how LangGraph
  conditional edges and the loop counter work here.
- [x] **3.5** Add the `escalate` node that builds a Jira ticket DRAFT (summary,
  error details, docs already checked from chunk metadata, suggested next
  steps) — but does NOT call Jira yet. Print the draft.

## Phase 4 — Jira tool
- [x] **4.1** Thin Jira REST wrapper: `create_issue(payload)` hitting
  `POST /rest/api/3/issue`. Respect `DRY_RUN` (log the payload instead of calling).
  Auth from env vars. Standalone-testable against a free Jira Cloud instance.
- [x] **4.2** Connect the `escalate` node to the Jira wrapper (still DRY_RUN by
  default). Confirm the full flow: unanswerable question → user opts to escalate
  (or confidence/attempts trigger it) → drafted → (dry-run) ticket logged.

## Phase 5 — Evals & observability (keep small)
- [x] **5.1** Small eval set: ~10–15 real AWS CodeBuild/CodePipeline questions with
  notes on expected relevant docs. Script to measure whether retrieval returns
  relevant chunks and whether the agent escalates when it should. Save results
  as a table for the README.
- [x] **5.2** Wire LangSmith tracing so I can see each graph step, and CloudWatch-
  friendly structured logging (JSON to stdout is enough).

## Phase 6 — Demo UI, deploy & CI/CD
- [x] **6.1** Minimal demo interface: a CLI entrypoint for dev, plus a small UI
  for the public demo (e.g., Streamlit + a tiny chat page) that
  supports the mid-conversation "resolved / open a ticket?" prompt. Jira forced
  to DRY_RUN/mock in demo mode.
- [x] **6.2** Containerize the agent (Dockerfile). Keep ingestion separate from
  serving (script run locally or as a Lambda later).
- [x] **6.3** GitHub Actions: build image → push to ECR. (Deploy step can start
  manual.)
- [x] **6.4** Deploy to ECS Fargate; secrets from AWS Secrets Manager; verify the
  public demo runs with Jira in dry-run/mock mode.
- [x] **6.5** README polish: architecture diagram, setup steps, eval results
  table, AWS docs attribution/license note (CC BY-SA 4.0). Resume-ready.

---

## Backlog / maybe-later
- Add IAM docs to the corpus if evals show permission-type gaps.
- Add a reranker to improve retrieval quality.
- Add more CI/CD docs (Step Functions, Lambda, EventBridge, S3, CloudFormation).
- Swap Pinecone → OpenSearch Serverless / pgvector if I want an all-AWS stack.
- Schedule ingestion via EventBridge (manual re-runs are fine for the demo).
- Nicer demo UI / streaming responses.
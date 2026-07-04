# CLAUDE.md — AWS Support Agent

Standing rules for this project. Read `project_summary.md` for full context and
`tasks.md` for the task list. Keep this file lean; update it when a rule changes.

## What this project is (one line)
An agentic RAG assistant that answers AWS CI/CD questions from AWS docs and
escalates unresolved issues by drafting a Jira ticket. Built to learn
RAG + agents + MLOps on AWS. Learning is a goal, not just a working repo.

## Working style (most important)
- **Do only the current task.** Do not start the next task or refactor unrelated
  code. When the task is done, stop and summarize.
- **Plan before code.** For any non-trivial task, first explain your approach and
  the key concepts (RAG / LangGraph / AWS) involved, then WAIT for my approval
  before writing code. Use plan mode.
- **Explain as you go.** After writing code, give a short "why it works" note and
  comment any non-obvious AWS / LangChain / LangGraph call. I am learning — favor
  clarity over cleverness.
- **Prefer simple, readable code.** Small functions, clear names, minimal
  abstraction. If a simpler version exists, use it. If I can't understand it, it's
  too complex — simplify.
- **Ask, don't assume.** If a task is ambiguous or a design choice matters, ask me
  rather than guessing.
- **I commit, not you.** Do not run `git commit` or `git push`. You may stage and
  draft a commit message for me to edit; I review every diff before committing.

## Tech stack
- Python 3.11+
- LangGraph (agent state machine) + LangChain (RAG plumbing)
- Pinecone (vector DB, serverless on AWS)
- Amazon Bedrock (Claude models) via `langchain-aws` for LLM + embeddings
- Jira Cloud REST API (ticket creation, as a tool)
- AWS deploy target: ECS Fargate (container); CI/CD via GitHub Actions + ECR
- Observability: CloudWatch + LangSmith (tracing/evals)

## Commands
(Fill in as they get created — this section is high value, keep it current.)
- Install: `pip install -r requirements.txt`
- Run ingestion: `python -m src.ingest`
- Run agent locally: `python -m src.app`
- Run tests: `pytest`
- Lint/format: `ruff check .` and `ruff format .`

## Project layout
(Update as structure grows.)
- `src/ingest/` — clone AWS docs, chunk, embed, upsert to Pinecone
- `src/agent/` — LangGraph graph, nodes, state
- `src/tools/` — Jira tool and other tool calls
- `src/evals/` — retrieval + agent eval scripts
- `tests/` — pytest tests

## Data source (important — non-obvious)
AWS retired the `awsdocs` GitHub repos and stripped the doc content from the
default (`archived`) branch. The markdown still exists in git history in the
commit **before** the "delete content directory" archival commit. Ingestion must
clone full history and check out that pre-archival commit to get `doc_source/*.md`.
Starting corpus: `aws-codebuild-user-guide`, `aws-codepipeline-user-guide`.
Content is CC BY-SA 4.0 — attribute AWS; do not commit raw doc text to this repo.

## Safety / secrets (never violate)
- No secrets in code. All keys (Pinecone, Jira, AWS, LangSmith) come from env vars.
- `.env` is gitignored. Never print or commit key values.
- **Jira ticket creation is a real side effect.** Always gate it behind a
  `DRY_RUN` env flag that defaults to true. In dry-run, log "would create ticket…"
  instead of calling the Jira API. Never wire it to auto-create without a
  confirmation step in the graph.
- Do not add new dependencies without telling me what and why.

## Testing
- Every piece should have a simple way to run/verify it.
- For RAG/agent behavior, prefer small eval scripts (does retrieval return
  relevant chunks? does the graph escalate when confidence is low?) over abstract
  unit tests where that's more meaningful.
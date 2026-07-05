# CLAUDE.md — AWS MLOps Support Agent

Agentic RAG assistant: answers AWS CI/CD questions from AWS docs, escalates unresolved
issues via Jira ticket draft. Goal = learning (RAG, LangGraph, MLOps), not just shipping.
Context: `project_summary.md`. Tasks: `tasks.md` — do ONE task at a time.

## Rules
- Do only the current task. No refactoring unrelated code. Done → stop, summarize.
- Plan first: explain approach + key concepts, WAIT for approval before coding.
- Simple, readable code. Short "why it works" note after; comment non-obvious
  AWS/LangChain/LangGraph calls. If ambiguous, ask — don't guess.
- Never `git commit`/`push`. Stage + draft commit message only; I commit.
- No new dependencies without asking.

## Stack
Python 3.11+ · LangGraph + LangChain · Pinecone serverless · OpenAI via
`langchain_openai` (LLM + embeddings) · Jira Cloud REST API · ECS Fargate ·
GitHub Actions + ECR · CloudWatch + LangSmith

## Commands
- Install: `uv sync`
- Ingest: `uv run python -m src.ingest`
- Agent: `uv run python -m src.app`
- Tests: `uv run pytest`
- Lint: `uv run ruff check .` / `uv run ruff format .`

## Data source (non-obvious)
`awsdocs` repos are archived with content stripped from the default branch. Clone
full history, check out the commit BEFORE the "delete content directory" commit to
get `doc_source/*.md`. Corpus: `aws-codebuild-user-guide`, `aws-codepipeline-user-guide`.
CC BY-SA 4.0: attribute AWS, never commit raw doc text.

## Safety
- Secrets only from env vars; `.env` gitignored; never print/commit key values.
- Jira creation gated by `DRY_RUN` (default true) — log payload instead of calling.
  Never auto-create without a confirmation step in the graph.

## Testing
Everything runnable/verifiable in isolation. Prefer small eval scripts over
abstract unit tests for RAG/agent behavior.

## Progress
After each task, append 1–2 lines to `progress.md` (built / changed in review / why).
Read `progress.md` at session start.
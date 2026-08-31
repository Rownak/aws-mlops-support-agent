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

## Safety
- Secrets only from env vars; `.env` gitignored; never print/commit key values.
- Jira creation gated by `DRY_RUN` (default true) — log payload instead of calling.
  Never auto-create without a confirmation step in the graph.

## Testing
Everything runnable/verifiable in isolation. Prefer small eval scripts over
abstract unit tests for RAG/agent behavior.
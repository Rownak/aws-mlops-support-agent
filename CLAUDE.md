# CLAUDE.md — RAG pipeline monorepo

`rag_core` is a corpus-agnostic RAG engine; two apps build on it —
`aws_mlops_support_agent` and `rag_bench_eval`.

**Current work: `packages/rag_bench_eval`** — retrieval-optimization benchmark on
BEIR/NFCorpus (BM25, dense, HyDE, multi-query, RRF, reranking), scored by nDCG@10.
Context: `packages\rag_bench_eval\design_summary.md`
Tasks: `packages\rag_bench_eval\tasks_v3.*.md`

## Rules
- Do only the current task or list of tasks instructed. No refactoring unrelated code. Done → stop, summarize.
- Plan first: explain approach + key concepts, WAIT for approval before coding.
- Simple, readable code. Short "why it works" note after; If ambiguous, ask — don't guess.
- `rag_core` changes stay additive: the AWS agent must keep working,
  `uv run pytest` green before every commit message.
- Never `git commit`/`push`. Stage + draft commit message only; I commit.
- No new dependencies without asking.

## Safety
Secrets only from env vars; `.env` gitignored; never print/commit key values.
Benchmark data (`data/beir/`) is CC BY-SA — fetched at run time, never committed.

## Testing
Runnable/verifiable in isolation. Prefer small eval scripts over abstract unit
tests — except metrics, which get real tests against hand-computed rankings:
a wrong nDCG invalidates every number.

## Progress
After each phase or when instructed, add 2-3 concise lines to
`packages/rag_bench_eval/progress.md`.

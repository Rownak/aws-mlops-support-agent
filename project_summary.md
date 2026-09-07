# Project Summary — RAG monorepo

## One-paragraph description
A RAG engine and the applications built on it. `rag_core` is a corpus-agnostic
retrieval-and-generation engine: ingestion, chunking, embeddings, vector stores,
retrieval techniques, and evaluation metrics. `rag_bench_eval` measures which
retrieval technique actually works, scoring them on labelled IR datasets so
engine changes are driven by numbers rather than intuition. `aws_mlops_support_agent`
is the main application — an agentic assistant that answers questions from
documentation and escalates to a Jira ticket when it can't. The engine is shared;
the benchmark tunes it; the agent ships it.

## Why it's structured this way
The engine exists separately because two things need it: an application that
must keep working, and a benchmark that must be free to experiment. Keeping
retrieval techniques in `rag_core` means a technique proven on a public dataset
is the same code the agent runs — no reimplementation between measuring and
shipping.

## Packages

| Package | Role |
|---|---|
| `rag_core` | The engine. Corpus-agnostic: ingestion, chunking, embeddings, vector stores (Pinecone, Qdrant, FAISS), retrieval techniques behind one `Retriever` protocol, and IR metrics. Changes here stay additive — the agent must keep working. |
| `rag_bench_eval` | The benchmark. Runs retrieval techniques over BEIR datasets (NFCorpus, CQADupStack Programmers) and scores them by nDCG@10 against human relevance labels. Answers "which technique, at what cost." |
| `aws_mlops_support_agent` | The application. LangGraph agent over AWS CI/CD documentation with Jira escalation and a Streamlit UI. |

## How they fit together
```
rag_bench_eval  ──measures──>  rag_core  <──uses──  aws_mlops_support_agent
                                  ^                 
                                  └── one Retriever protocol, shared by both
```

The `Retriever` protocol (`search(query, k) -> list[SearchResult]`) is the seam.
A technique benchmarked in `rag_bench_eval` and a technique running in the agent
are the same implementation, so a benchmark result transfers by configuration
rather than by rewrite.

## Current state
- **`rag_core`:** ingestion and retrieval working; BM25, dense, RRF fusion, and
  bi-/cross-encoder reranking implemented behind the protocol.
- **`rag_bench_eval`:** five techniques scored on two datasets. Dense retrieval
  beats BM25 on both; reranking wins on one corpus and loses on the other, so the
  winner does not transfer automatically. See `packages/rag_bench_eval/README.md`.
- **`aws_mlops_support_agent`:** agent flow (retrieve → answer → confirm →
  escalate) working; the benchmark winner is not yet wired in.
- **Next:** wrap Pinecone behind the `Retriever` protocol so the agent can use
  composed retrieval, then verify the winner on the AWS corpus.

## Tech stack
- **Orchestration:** LangGraph (state machine with a loop counter and a
  user-confirmation step before escalating).
- **RAG plumbing:** LangChain — loaders, splitters, retrievers, vector store
  integrations.
- **Vector DB:** Pinecone serverless; FAISS/Qdrant supported in `rag_core`.
- **LLM + embeddings:** OpenAI via `langchain_openai`; Ollama locally for
  benchmark runs. Embedding model stays consistent between ingestion and query.
- **Benchmark:** BEIR datasets, nDCG@10 primary, metrics unit-tested against
  hand-computed rankings.
- **Compute / CI:** ECS Fargate, GitHub Actions → ECR → Fargate.
- **Secrets:** AWS Secrets Manager in prod; `.env` locally, never committed.

## Non-goals
- `rag_core` never imports from an application or from `rag_bench_eval`.
- Not a general BEIR harness — the protocol generalizes, the benchmark package
  does not.
- Not multi-tenant SaaS; auth distinguishes admin vs. member, not organizations.

## Per-package detail
- `packages/aws_mlops_support_agent/project_summary.md` — the agent's own spec.
- `packages/rag_bench_eval/README.md` — benchmark results and findings.
- `packages/rag_bench_eval/design.md` — retrieval design and rationale.
- `claude/docs/project_summary.md` — the team-docs product direction (a
  generalization of the AWS agent; not yet built).

## How I'm working with Claude Code
Small tasks, one at a time. Claude Code proposes a plan, I approve, it writes
simple code with explanation, I review/test/adjust, then I commit manually. See
`CLAUDE.md` for the standing rules.

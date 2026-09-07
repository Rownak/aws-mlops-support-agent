# Project Summary — AWS Support Agent

## One-paragraph description
A production-style agentic AI assistant (portfolio project). A user describes an
AWS CI/CD problem or question. The agent runs RAG over official AWS documentation
(starting with CodeBuild and CodePipeline) to try to answer it. If the issue is
not resolved after reasonable attempts — or retrieval confidence is low — the
agent drafts and creates a Jira ticket (summary, error details, docs already
checked, suggested next steps) for a "developer team." Deployed on AWS with
CI/CD, logging, and basic evals, with a public demo and GitHub repo.

## Why I'm building it
1. Portfolio piece for my resume, inspired by patterns from my recent data scientist role.
2. Hands-on: LLM RAG, agentic AI (state machines), and MLOps on AWS.

## Core flow (agent state machine)
retrieve → answer → confirm resolution → escalate

- **retrieve**: embed the user's question, query Pinecone for relevant AWS doc chunks.
- **answer**: LLM (Bedrock Claude) generates an answer grounded in retrieved chunks,
  with citations (which doc/section).
- **confirm resolution**: check whether the issue is resolved. Track attempts and
  retrieval confidence. Loop back to retrieve if it's worth another try.
- **escalate**: if unresolved after N attempts OR confidence is low OR
  the user is not satisfied with the answer and wants to open a ticket,
  draft a Jira ticket and (unless DRY_RUN) create it via the Jira API.


## Scope decisions already made
- **Focus:** CI/CD & pipeline automation (not model-training/SageMaker-heavy).
- **Starting corpus (minimal, 2 docs):** `aws-codebuild-user-guide`,
  `aws-codepipeline-user-guide`. More CI/CD docs (Step Functions, Lambda,
  EventBridge, S3, CloudFormation) added later once the pipeline works.
- **IAM** is the most likely "next add" if evals show permission-type gaps.

## Data source detail (important)
The `awsdocs` GitHub repos are archived and their content was removed from the
default branch. The doc markdown lives in git history in the commit before the
archival ("delete content directory") commit. Ingestion clones full history and
checks out that pre-archival commit to recover `doc_source/*.md`. Content is
frozen ~2023 — fine for a portfolio demo. License: CC BY-SA 4.0 (attribute AWS;
don't redistribute raw doc text in the repo — pull it at build time).

## Tech stack
- **Orchestration:** LangGraph (state machine) — nodes + conditional edges + a loop
  counter for "reasonable attempts before escalating." plus a user-input step where the agent asks whether the user is satisfied or wants to open a Jira ticket, routing to escalate on request.

- **RAG plumbing:** LangChain (loaders, splitters, retriever, Pinecone integration).
- **Vector DB:** Pinecone serverless (on AWS).
- **LLM + embeddings:** OpenAi Models via `langchain_openai`. Keep the
  embedding model consistent between ingestion and query.
- **Tool call:** Jira Cloud REST API (`POST /rest/api/3/issue`) via a thin custom
  wrapper — clearer than an off-the-shelf toolkit for learning tool-calling.
- **Compute:** ECS Fargate (container) for the agent; ingestion as a separate
  script/Lambda (optionally scheduled via EventBridge).
- **CI/CD:** GitHub Actions → build container → push to ECR → deploy to Fargate.
- **Secrets:** AWS Secrets Manager in prod; `.env` locally.
- **Observability/evals:** CloudWatch logs; LangSmith for agent-step tracing and
  eval runs.

## Deliverables
- Public demo (with Jira in DRY_RUN / mock mode so visitors can't spam the board).
- Clean GitHub repo with README, architecture notes, and eval results.

## Non-goals (for now)
- Not covering model training / SageMaker deeply.
- Not indexing the whole AWS doc set — deliberately minimal corpus.
- Not multi-user auth / accounts — it's a demo.

## How I'm working with Claude Code
Small tasks, one at a time (see `tasks.md`). For each: Claude Code proposes a plan,
I approve, it writes simple code with explanation, I review/test/adjust, then I
commit manually. See `CLAUDE.md` for the standing rules.
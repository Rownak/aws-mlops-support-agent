# RAG Retrieval Benchmark & Evaluation

## Overview

A benchmark harness that measures retrieval quality on labelled IR datasets.
Each technique is defined as a named pipeline in `benchmark.yaml`, run over
every query in a dataset, and scored against human relevance judgements. Runs
are written to `results/runs/*.json` and compared in `results/results.md`.

Retrieval techniques live in `rag_core` (reusable); dataset loading, config and
experiment workflow live here.

## Why This Project?

Retrieval quality sets the ceiling on any RAG system — if the right document
never reaches the context window, no amount of prompting recovers it. Yet
retrieval changes are usually judged by eyeballing a handful of queries, which
cannot distinguish a real gain from noise.

This project replaces that with measurement: every technique is scored on the
same queries against the same labels, so "hybrid beats dense" is a number, not
an impression. The BM25 baseline is validated against a published figure first,
which is what makes the rest of the numbers trustworthy.

## What We Evaluate

Ranking quality, not answer quality — no LLM generation is scored here. For a
technique the question is whether relevant documents appear in the top-10, and
how far up. Cost is tracked alongside quality (`mean_latency_ms`, `llm_calls`),
since a small nDCG gain bought with a 20× latency increase is rarely worth it.

## Datasets

| Dataset | Docs | Test queries | Labels | Notes |
|---|---|---|---|---|
| BEIR/NFCorpus | 3,633 | 323 | graded 0/1/2 | Medical. Fetched at run time. |
| BEIR/CQADupStack Programmers | — | 876 | binary | Technical Q&A. Local BEIR-format files. |

NFCorpus is the primary dataset: it is small enough to sweep quickly and has a
published BM25 nDCG@10 ≈ 0.32, used as a correctness gate. CQADupStack
Programmers is closer in domain to developer support, showing whether findings
transfer or are corpus-specific.

Benchmark data is CC BY-SA and never committed — see `data/` handling in
`settings.py`.

## Retrieval Techniques

Every technique implements one `Retriever` protocol
(`search(query, k) -> list[SearchResult]`), so they compose by nesting.

| Technique | How it works |
|---|---|
| **BM25** | Lexical. Term overlap weighted by frequency and document length. Untuned tokenizer, deliberately a plain baseline. |
| **Dense** | Semantic. Embeds query and documents into one vector space, ranks by cosine similarity. Matches paraphrases BM25 misses. |
| **RRF (hybrid)** | Fuses rankings from several retrievers by reciprocal rank — `Σ 1/(k + rank)`. Uses rank position, so incomparable score scales never need normalizing. |
| **Bi-encoder rerank** | Re-scores a wide candidate pool with an embedding model. Pointed at the same model as dense, it is a sanity check; pointed at a stronger one, an upgrade. |
| **Cross-encoder rerank** | Re-scores each (query, document) pair jointly, so the model reads both together. Most accurate, far too slow for a full corpus — hence the two-stage retrieve-then-rerank shape. |
| **HyDE** *(planned)* | Generates a hypothetical answer, retrieves against that instead of the question. |
| **Multi-query** *(planned)* | Generates query variants, retrieves each, fuses with RRF. |

Reranking pipelines request `candidate_k` documents from their inner retriever
and return `top_k` — a reranker can only reorder what it is given.

## Evaluation Metrics

| Metric | What it answers |
|---|---|
| **nDCG@10** *(primary)* | Are relevant docs ranked high in the top 10? Rewards position and uses graded labels. |
| **recall@100** | Does a wide candidate pool contain the relevant docs at all? The ceiling reranking can reach. |
| **MRR@10** | How far down is the *first* relevant doc? |
| **precision@10** | What fraction of the top 10 is relevant? |

Unjudged documents count as non-relevant, matching standard BEIR/trec_eval
behaviour. Metrics are unit-tested against hand-computed rankings — a wrong
metric would invalidate every number in this project.

## Project Structure

```
packages/rag_bench_eval/
  benchmark.yaml            resources, pipelines, sweep list, metrics
  design.md                 full design + rationale
  design_summary.md         condensed reference for implementation
  tasks_v3.0.md             phased task list
  src/rag_bench_eval/
    cli.py                  download / list / run / report
    config.py               benchmark.yaml loader
    resources.py            get_embeddings / get_reranker — dict-cached by name
    retrievers.py           pipeline config -> Retriever, wired to the caches
    index_cache.py          in-process index reuse across a sweep
    embedding_cache.py      corpus vectors persisted to disk
    evaluator.py            run orchestration + run JSON
    report.py               runs/*.json -> results.md
    settings.py             paths
    datasets/               types, nfcorpus, cqadupstack_programmers
  results/
    runs/*.json             one file per run, with per-query detail
    results.md              comparison table
  tests/
```

Retrieval and scoring live in `rag_core`:

```
packages/rag_core/src/rag_core/
  retriever/    base (protocol), bm25, dense, fusion, rerank, factory
  evals/        metrics, ir_runner
```

## Results

NFCorpus, full 323-query test set:

| experiment | ndcg@10 | recall@100 | mrr@10 | precision@10 |
|---|---|---|---|---|
| hybrid_cross_encoder | 0.3554 | 0.2559 | 0.5816 | 0.2517 |
| hybrid | 0.3413 | 0.3019 | 0.5588 | 0.2486 |
| bi_encoder_rerank | 0.3408 | 0.2503 | 0.5349 | 0.2523 |
| dense | 0.3408 | 0.3040 | 0.5349 | 0.2523 |
| bm25 | 0.3052 | 0.2385 | 0.5140 | 0.2155 |

CQADupStack Programmers, full 876-query test set:

| experiment | ndcg@10 | recall@100 | mrr@10 | precision@10 |
|---|---|---|---|---|
| bi_encoder_rerank | 0.4279 | 0.7279 | 0.4202 | 0.0776 |
| dense | 0.4279 | 0.7940 | 0.4202 | 0.0776 |
| hybrid_cross_encoder | 0.3932 | 0.6937 | 0.3916 | 0.0707 |
| hybrid | 0.3651 | 0.7624 | 0.3645 | 0.0655 |
| bm25 | 0.2680 | 0.5017 | 0.2712 | 0.0453 |

Full tables with latency and LLM-call cost: `results/results.md`,
`results/results_cqadupstack_programmers.md`.

## Findings

- **Dense beats BM25 on both datasets** — by 0.036 nDCG@10 on NFCorpus, by 0.160 on
  CQADupStack. The only result that held its direction and rough magnitude across
  corpora.
- **The winner does not transfer.** Cross-encoder reranking tops NFCorpus (0.3554)
  but *loses* to plain dense on CQADupStack (0.3932 vs 0.4279). A technique ranked
  on one corpus should not be assumed to win on another.
- **Bi-encoder reranking is a no-op** when pointed at the same embedding model as
  first-stage dense — it ties `dense` to four decimals on both datasets. It
  re-ranks the same signal rather than adding one, which is the intended sanity
  check that the harness is wired correctly.
- **RRF hybrid never clearly wins.** It edges past its parents on NFCorpus (+0.0005
  over dense) and lands well below dense on CQADupStack. Fusion pays off only when
  lexical and dense fail differently; here dense dominates too consistently.
- **Reranking always costs recall@100.** Every reranked pipeline scores below its
  own first stage on recall — cross-encoder drops NFCorpus recall from 0.3040 to
  0.2559, CQADupStack from 0.7940 to 0.6937. Reranking reorders a truncated
  candidate pool, so it cannot recover documents the first stage missed.
- **Latency scales with reranking, not quality.** Cross-encoder rerank runs ~12x
  slower than dense on NFCorpus for +0.015 nDCG, and ~8x slower on CQADupStack for
  −0.035. For a live agent, dense alone is the better default on this evidence.

## Quick Start

```bash
uv sync
uv run rag-bench-eval download                 # fetch + cache NFCorpus

uv run rag-bench-eval list                      # pipelines in benchmark.yaml
uv run rag-bench-eval run --experiment bm25 --limit 20   # smoke test
uv run rag-bench-eval run --all                  # full sweep, all pipelines
uv run rag-bench-eval report                     # regenerate results/results.md
```

`--dataset cqadupstack_programmers` runs against the second dataset instead of
NFCorpus. Ollama must be running locally for any `dense`-based pipeline
(`nomic-embed-text` pulled). First cross-encoder run downloads its model (~90MB).

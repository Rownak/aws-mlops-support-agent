# design_summary — `rag-bench-eval`

Reference for implementation tasks. Rationale lives in `design.md`; this file is
structures and rules only.

Import `rag_bench_eval`, distribution `rag-bench-eval`. Benchmark: BEIR/NFCorpus
(3,633 docs, 323 test queries, graded 0/1/2 qrels). Primary metric nDCG@10.
BM25 baseline target ≈ 0.32.

---

## Contract

```python
@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    document: Document
    score: float
    score_type: str      # "bm25" | "cosine" | "rrf" | "rerank_logit"

class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[SearchResult]: ...
```

Concrete: `BM25Retriever`, `DenseRetriever`, `PineconeRetriever`,
`HyDERetriever`, `MultiQueryRetriever`, `RRFRetriever`, `RerankingRetriever`.
No `Leaf`/`Decorator`/`Combinator` base classes — protocol + concretes only.

Rules:
- `top_k` is a request: a parent passes the depth it needs to `search(q, k)`;
  a node's configured `top_k` applies only when nothing above overrides.
- `min_score` only on the outermost node; it checks `score_type`.
- Persist `doc_id` + `score`; `document` never reaches disk.

## Config (`benchmark.yaml`, one file)

```yaml
embeddings: {default: {provider, model, base_url}, ...}
llms:       {default: {...}, query_expansion: {provider, model, temperature: 0}}
rerankers:  {bi_encoder: {provider, embeddings}, cross_encoder: {provider, model}}

retrieval:
  active: production
  pipelines:
    bm25:    {type: bm25, k1: 0.9, b: 0.4, top_k: 10}
    dense:   {type: dense, embeddings: default, metric: cosine, top_k: 10}
    hyde_dense:
      type: hyde
      llm: query_expansion
      include_original: true
      inner: {type: dense, embeddings: default, metric: cosine}
      top_k: 10
    multiquery_dense:
      type: multi_query
      llm: query_expansion
      num_queries: 4
      rrf_k: 60
      inner: {type: dense, embeddings: default, metric: cosine}
      top_k: 10
    hybrid_cross_encoder:
      type: rerank
      reranker: cross_encoder
      candidate_k: 50
      top_k: 10
      inner:
        type: rrf
        rrf_k: 60
        retrievers:
          - {type: bm25, k1: 0.9, b: 0.4}
          - {type: dense, embeddings: default, metric: cosine}

sweep: [bm25, dense, hyde_dense, multiquery_dense, hybrid, hybrid_cross_encoder]
evaluation: {metrics: [ndcg@10, recall@100, mrr@10, precision@10], primary: ndcg@10}
```

- Children nest inline. No cross-references, no `indexes:` block — each node
  owns its settings. Duplication is accepted.
- `build_retriever(cfg, resources)` = recursive `if cfg.type == ...` dispatch;
  unknown type raises at load.
- Resource getters are dict caches: `get_embeddings/get_llm/get_reranker(name)`.
- Index construction cached on its settings (embeddings+metric, or k1/b) so
  dense is embedded once per sweep. Runtime only, not config.
- `embeddings:` accepts a bare name, or `{query: …, passage: …}` for
  asymmetric encoders (DPR).

## Layout

```
benchmark.yaml
src/rag_bench_eval/
  settings.py             paths, cache dirs
  datasets/
    types.py              Doc, Query, Qrels — no network imports
    nfcorpus.py           download_nfcorpus() + load_nfcorpus()
  evaluator.py            Retriever × queries × qrels → RunResult
  llm_cache.py            _key / load_cache / save_cache  (phase 5)
  report.py               RunResult[] → comparison table
  cli.py                  download / list / run / report
results/runs/*.json, results/results.md
tests/
```

`rag_core` additions: `retriever/{base,bm25,dense,expansion,fusion,rerank,
factory}.py`, `evals/{metrics,ir_runner}.py`, `config/{retrieval,resources}.py`,
`pipeline.py` (`_vectorstore()` → `_retriever()`).

## Run JSON

```json
{"experiment": "", "dataset": "nfcorpus", "timestamp": "", "config": {},
 "metrics": {"ndcg@10": 0.0, "recall@100": 0.0, "mrr@10": 0.0},
 "per_query": [{"query_id": "", "ndcg@10": 0.0, "latency_ms": 0,
                "retrieved": ["doc_id"]}],
 "llm_calls": 0}
```

`llm_calls` counts real calls, not cache hits. `git_commit` deferred.

LLM cache: flat `{hash: value}` in `.cache/query_expansion.json`, key =
`_key(model, prompt, query)`, saved after each call.

## Experiments

1 bm25 · 2 dense · 3 hyde · 4 multiquery+rrf · 5 bi-encoder rerank ·
6 cross-encoder rerank · 7 combos (chosen after 1–6).

## Build order

0 scaffold · **1 end-to-end: load → protocol → BM25 → nDCG@10 (gate ≈0.32)** ·
2 embeddings + dense + index cache · 3 extract runner to `rag_core.evals` ·
4 RRF + rerankers · 5 llms + HyDE + MultiQuery + cache · 6 combos ·
7 Pinecone cutover · 8 fold winner into AWS agent.

Deps needing approval: `rank_bm25` (phase 1), `numpy` (phase 2).
Not used: `beir`, `pytrec_eval`.

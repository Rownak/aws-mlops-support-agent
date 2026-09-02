# Design — `rag-bench-eval`

Evaluate RAG retrieval optimization techniques on BEIR/NFCorpus, built on `rag_core`.

Import name `rag_bench_eval`, distribution name `rag-bench-eval`
(matching the `rag-core` / `rag_core` convention).

**Status:** design draft, iteration 1. No code written.

---

## 1. Goal

Measure how much each retrieval technique actually helps, on a public benchmark
with graded relevance labels — then enable whatever wins in the AWS agent.

NFCorpus fits: 3,633 medical documents, 323 layperson test queries, graded
(0/1/2) qrels. Small enough to sweep in minutes; the query/document vocabulary
gap is wide enough that query expansion and reranking have something to do.
Published BM25 nDCG@10 ≈ 0.32 gives us a number to validate our harness against.

**Non-goals:** not a general BEIR harness (one dataset), not answer-quality
evaluation (retrieval only), not every possible combination.

---

## 2. Architecture

```
                        RagCore
                           │
                           ▼
                    Retriever protocol
                           │
         ┌─────────────────┼──────────────────┐
         │                 │                  │
      Pinecone            BM25              Dense
         │                 │                  │
         └──────────┬──────┴─────────┬────────┘
                    │                │
               Query transforms    Fusion
                 HyDE              RRF
                 MultiQuery         │
                    │               │
                    └───────┬───────┘
                            │
                         Reranker
                            │
                            ▼
                    ranked documents
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
           Production RAG         BEIR evaluator
                 │
                 ▼
             Generator
```

Everything above the fork is `rag_core`. The BEIR evaluator's *dataset* and
*experiment workflow* are the new package; the metrics it calls are `rag_core`.

---

## 3. Four decisions

### 3.1 `RagCore` stops choosing storage

```
before:  RagCore → _vectorstore() → PineconeStore
after:   RagCore → _retriever()   → build_retriever(config) → Retriever
```

`Retriever` is one structural protocol:

```python
@dataclass(frozen=True)
class SearchResult:
    doc_id: str          # stable corpus id — what qrels are matched against
    document: Document
    score: float
    score_type: str      # "bm25" | "cosine" | "rrf" | "rerank_logit"

class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[SearchResult]: ...
```

Pinecone, BM25 and in-memory dense all satisfy it. Techniques wrap it —
`HyDE(inner)`, `RRF([a, b])`, `Rerank(inner, reranker)` — so every experiment is
one nested expression, and the production agent and the benchmark run the
identical code path.

This is the load-bearing decision. Without it, BM25 (no vectors) and a
seven-configuration sweep (no networked index per config) have nowhere to live.

**Why a dataclass, not `tuple[Document, float]`.** `r.score` reads better than
`r[1]`, and the current code already pays for the tuple — `retrieve.py` unpacks
and rebuilds it twice, `evals/runner.py` unpacks to discard half. More
substantially, two fields deserve to be part of the contract rather than
conventions:

- **`doc_id`** is the stable corpus identifier scoring matches against qrels.
  Today the equivalent is `Document.metadata["source"]`, reached into by
  `evals/runner.py`'s `chunk_label()`. It is part of the contract, so it is a
  field.
- **`score_type`** names the scale, because scores are *not* commensurable
  across retrievers: BM25 is unbounded, cosine is [-1, 1], a cross-encoder logit
  is neither, and an RRF score is a rank artifact. Ordering within one
  retriever's output is always meaningful; the absolute value is only meaningful
  against a matching scale. This is what makes the `min_score` trap
  (`retrieve.py`'s own warning, and §3.2) checkable instead of silent — a
  threshold node can assert `score_type == "cosine"` and refuse otherwise.
  *Optional:* drop the field if the assertion is not wanted, but then the
  comparability rule has to live in documentation instead.

**`document` stays in memory; only `doc_id` and `score` are persisted.** Run
JSONs record ids, never document text — otherwise every result file carries a
copy of the corpus (§4).

### 3.2 One active retrieval pipeline, selected by name

**Every node is a retriever**, and composition is nesting: a node holds its
children inline under `inner` (one) or `retrievers` (several). Query expansion
is not a modifier on a retriever — it *is* a retriever, which is what lets HyDE,
fusion and reranking nest in any order without new config vocabulary.

**Each retriever owns its own implementation settings.** There is no top-level
`indexes:` block: a Pinecone connection, a NumPy embedding matrix and BM25 term
statistics share no fields, so grouping them under one vocabulary would name a
category that does not exist and add a lookup for nothing. `k1`/`b` live on the
BM25 node, `metric` on the dense node. If two pipelines ever need to share a
genuinely *configurable* index, extract `indexes:` then.

The whole builder is a recursive dispatch on `type`:

```python
def build_retriever(cfg, resources):
    if cfg.type == "bm25":
        return BM25Retriever(corpus, k1=cfg.k1, b=cfg.b, top_k=cfg.top_k)
    if cfg.type == "dense":
        return DenseRetriever(corpus, embeddings=resources.embeddings(cfg.embeddings),
                              metric=cfg.metric, top_k=cfg.top_k)
    if cfg.type == "hyde":
        return HyDERetriever(build_retriever(cfg.inner, resources),
                             llm=resources.llm(cfg.llm), ...)
    if cfg.type == "rrf":
        return RRFRetriever([build_retriever(c, resources) for c in cfg.retrievers], ...)
    if cfg.type == "rerank":
        return RerankingRetriever(build_retriever(cfg.inner, resources),
                                  reranker=resources.reranker(cfg.reranker), ...)
    raise ValueError(f"unknown retriever type: {cfg.type}")
```

That covers all seven planned pipelines. No name resolution between retrievers,
no graph — each pipeline is a self-contained tree, built fresh.

**But the expensive parts are built once.** `dense` appears in four pipelines
(`hyde`, `multiquery`, `hybrid`, `hybrid+rerank`), and embedding 3,633 documents
four times per sweep is minutes of pure waste. So the *construction* of an index
is cached on the settings that determine it — for dense, the embedding resource
plus the metric; for BM25, `k1`/`b` — and `build_retriever` returns retrievers
that share the one built artifact:

```python
matrix = index_cache.dense(embeddings="default", metric="cosine")  # built once
```

This is deliberately a **runtime cache, not a config vocabulary**. A shared
NumPy matrix is an optimization detail; making it a named config entity was what
`indexes:` got wrong. The cache key is the settings themselves, so two pipelines
that configure dense identically share automatically, and two that differ do not
— no name for anyone to get wrong.

The cache belongs to the experiment runner's process. `RagCore` in production
builds one pipeline against a live Pinecone index and needs none of it.

```yaml
retrieval:
  active: production          # the pipeline RagCore uses

  pipelines:
    production:               # the AWS agent: a live Pinecone index
      type: pinecone
      collection_name: aws-docs
      embeddings: default     # a resource reference (§3.3)
      top_k: 5
      min_score: 0.35         # only meaningful on the outermost node

    bm25:
      type: bm25
      k1: 0.9                 # settings live on the node that uses them
      b: 0.4
      top_k: 10

    dense:
      type: dense
      embeddings: default
      metric: cosine
      top_k: 10

    hyde_dense:
      type: hyde
      llm: query_expansion    # named LLM (§3.3), not the generation LLM
      include_original: true  # concatenate the real query with the hypothetical
      inner: {type: dense, embeddings: default, metric: cosine}
      top_k: 10

    multiquery_dense:
      type: multi_query
      llm: query_expansion
      num_queries: 4
      rrf_k: 60               # it fans out, so it merges — same RRF as below
      inner: {type: dense, embeddings: default, metric: cosine}
      top_k: 10

    hybrid_cross_encoder:
      type: rerank
      reranker: cross_encoder # a name from `rerankers:` (§3.3)
      candidate_k: 50
      top_k: 10
      inner:
        type: rrf
        rrf_k: 60
        retrievers:
          - {type: bm25,  k1: 0.9, b: 0.4}
          - {type: dense, embeddings: default, metric: cosine}
```

An experiment is then `retrieval.active: <name>` — one line, and the pipeline
block serializes into the run manifest as-is, which is what makes a result
reproducible.

**Why nesting rather than a `transform:` attribute.** The alternative —
`type: dense` plus a `transform: hyde` key — needs two composition mechanisms
and still cannot express expansion *above* a reranker. Nesting has one rule and
one dispatch table, and covers the experiment-7 combos we cannot predict yet.

Some duplication is accepted: `{type: dense, embeddings: default, metric: cosine}`
appears in four pipelines. That is the price of dropping cross-references, and it
is the right trade at this size — each pipeline reads top-to-bottom with nothing
to look up. It costs nothing at runtime either: identical settings hit the same
index-cache entry, so the embedding matrix is built once, and the model behind
`embeddings: default` is loaded once as a named resource (§3.3).

Two rules worth stating, because breaking them produces wrong numbers rather
than errors:

- **`top_k` is a request, not a fixed property.** `search(query, k)` is already
  parameterised, so a parent asks its child for the depth *it* needs; a node's
  configured `top_k` is what it uses when nothing above overrides. Without this,
  `hybrid_cross_encoder` is incoherent — `candidate_k: 50` cannot be satisfied by
  children that each return 10.
- **`min_score` belongs only on the outermost node.** It produces a *verdict*
  (the agent's escalate-or-answer branch), not a ranking, and a threshold applied
  before a reranker would drop documents the reranker was meant to rescue. It is
  also the one place an absolute score matters rather than an ordering, so it
  checks `score_type` (§3.1) and refuses a scale it was not tuned for — a
  reranker's logits are not normalized similarities (see
  `retriever/retrieve.py`'s own warning).

Unknown `type` values fail at config-load time, like every other config error —
that is `RagConfig._validate()`'s existing posture, not new machinery.

### 3.3 Providers become named resources

Today's config assumes exactly one embedding model and one LLM. The experiments
need several of each at once — a baseline dense encoder, DPR's question and
passage encoders, a late-chunking model, a reranker, a HyDE LLM, a multi-query
LLM. So the singular blocks become named maps, and pipeline nodes reference
entries by name.

```yaml
embeddings:
  default:
    provider: ollama
    model: nomic-embed-text
    base_url: "http://localhost:11434"

  dpr_question:
    provider: huggingface
    model: facebook/dpr-question_encoder-single-nq-base

  dpr_passage:
    provider: huggingface
    model: facebook/dpr-ctx_encoder-single-nq-base

llms:
  default:
    provider: ollama
    model: "llama3.1:8b"
    base_url: "http://localhost:11434"

  query_expansion:
    provider: ollama
    model: "llama3.1:8b"
    temperature: 0

rerankers:
  bi_encoder:
    provider: bi_encoder
    embeddings: default              # a resource reference, not a model string

  cross_encoder:
    provider: cross_encoder
    model: cross-encoder/ms-marco-MiniLM-L-6-v2
```

This is what makes `embeddings: default` and `llm: query_expansion` in §3.2
resolvable — without it those names point at nothing.

**Resolution is a dict cache, not a resource subsystem.** One function per map,
memoised by name:

```python
_embedding_cache: dict[str, Embeddings] = {}

def get_embeddings(name, cfg):
    if name not in _embedding_cache:
        _embedding_cache[name] = build_embeddings(cfg.embeddings[name])
    return _embedding_cache[name]
```

That is the whole mechanism. It gives lazy construction (an experiment using one
model does not pay to load four) and instance sharing (a dense retriever and a
bi-encoder reranker both naming `default` get the same object) as consequences
of caching, not as separate features. Same three lines for `get_llm` and
`get_reranker`.

**Why `rerankers:` stays its own map.** A bi-encoder reranker is an embedding
model and could arguably live in `embeddings:`, but a cross-encoder cannot: it
scores a (query, document) pair and returns one number, which is a different
interface. Folding it in would make `embeddings:` mean two incompatible things.
`provider` is the discriminator, matching the other maps and today's
`get_reranker()`; `type` is deliberately not reused, since in `pipelines:` it
means topology.

`bi_encoder` referencing an embedding resource is deliberate: pointed at the same
`default` the dense retriever uses, it re-scores with the identical model and
should show near-zero gain — a useful sanity check. Pointed at a stronger
resource, it is experiment 5.

Two things worth stating:

- **`default` is the conventional name**, used when a reference is omitted, so
  the AWS agent names nothing and keeps working.
- **`temperature: 0` on `query_expansion` is not incidental.** Query expansion
  is part of a measured pipeline; a sampling LLM makes a run unreproducible.
  Generation can stay warm.

**Validation keeps today's fail-fast posture, now per resource.**
`RagConfig._validate()` already collects every missing secret in one error, and
`missing_readiness()` already pings Ollama to confirm a model is actually
pulled. Neither is new machinery — the only changes are that the loop runs over
the maps instead of one block, and that an error names the *resource*
(`embeddings.dpr_question`) rather than the provider, or a config with four
HuggingFace models emits four identical messages.

Checks cover the resources the active config actually references — you cannot
ping a model you have not decided to build, and lazy construction is the point.
That check runs at load time, before any retrieval: a sweep that dies on
experiment 5 because a model was never downloaded has already spent the LLM
budget for experiments 1–4.

**The asymmetric-encoder problem.** `dpr_question` and `dpr_passage` are not
interchangeable — DPR encodes the query with one model and documents with
another. Today `get_embedding()` returns a single object used for both sides,
and a node configured with `embeddings: default` assumes symmetry. A DPR
retriever therefore needs two references:

```yaml
dpr:
  type: dense
  embeddings: {query: dpr_question, passage: dpr_passage}
  metric: dot
  top_k: 10
```

Accepting either a bare name (symmetric, the common case) or a `{query, passage}`
pair keeps every existing config untouched while making the asymmetric case
expressible. This also covers instruction-prefixed models (E5, BGE), which need
different prefixes per side for the same reason.


### 3.4 Evaluation metrics live in `rag_core`

nDCG, recall, precision and MRR are RAG concerns, not BEIR concerns. Putting
them in `rag_core.evals` means the AWS agent can run an annotated-data
evaluation stage with the same code, and the existing binary hit@k runner gains
graded-relevance company.

```yaml
evaluation:
  metrics: [ndcg@10, recall@100, mrr@10, precision@10]
  primary: ndcg@10
```

`rag_core` holds every RAG technique *and* every RAG evaluation. The new package
holds only what is specific to this dataset and this experiment.

*Where it lands, not when.* The metrics start local to the benchmark and move to
`rag_core.evals` in phase 3, once BM25 and dense have both used them — extracting
from two callers rather than guessing the shape from one (§6).

---

## 4. Package split

| Concern | Home |
|---|---|
| Retriever protocol, BM25, dense, HyDE, multi-query, RRF, rerankers | `rag_core.retriever` |
| Named resources, index construction, the `retrieval` config block | `rag_core` |
| nDCG / recall / precision / MRR, the metric-runner | `rag_core.evals` |
| NFCorpus download, cache, parse (corpus, queries, qrels) | `rag_bench_eval` |
| Benchmark config, sweep list, evaluator, results table | `rag_bench_eval` |

### `rag_core` — new and changed

```
retriever/
  base.py        SearchResult + Retriever protocol; adapter over the Pinecone path
  bm25.py        BM25Retriever — lexical, in-memory
  dense.py       DenseRetriever — in-memory embeddings + exact cosine
  expansion.py   HyDERetriever, MultiQueryRetriever — each wraps an inner one
  fusion.py      RRFRetriever — reciprocal rank fusion over n retrievers
  rerank.py      + bi-encoder; + RerankingRetriever wrapper
  factory.py     build_retriever(cfg, resources) — recursive dispatch on `type`
evals/
  metrics.py     ndcg_at_k, recall_at_k, precision_at_k, mrr_at_k
                 — all take ranked doc_ids, never Documents
  ir_runner.py   run a Retriever over a query set + qrels → scored result
config/
  retrieval.py   typed pipeline blocks for §3.2
  resources.py   get_embeddings / get_llm / get_reranker — dict-cached by name
  providers.py   EDIT: singular blocks become entries in those maps
pipeline.py      _vectorstore() → _retriever()
```

Additive. Existing `retrieve()` / `retrieve_scored()` behaviour is preserved
behind the `production` retriever.

### `rag_bench_eval`

```
benchmark.yaml            resources + pipelines + sweep + evaluation — one file
src/rag_bench_eval/
  settings.py             paths, cache dirs
  datasets/
    types.py              Doc, Query, Qrels — the contract, no network code
    nfcorpus.py           download_nfcorpus() + load_nfcorpus()
  evaluator.py            Retriever × queries × qrels → RunResult
  llm_cache.py            _key / load_cache / save_cache — phase 5, ~20 lines
  report.py               RunResult[] → comparison table
  cli.py                  download / list / run / report
results/
  runs/*.json             one per experiment run
  results.md              the comparison table
tests/
```

The dataset itself stays gitignored (CC BY-SA, fetched at run time); `results/`
is committed — the numbers are the deliverable.

**One config file, not a composed set.** Resources (§3.3) at the top, `pipelines`
(§3.2) below, then:

```yaml
sweep: [bm25, dense, hyde_dense, multiquery_dense, hybrid, hybrid_cross_encoder]
```

`--experiment <name>` runs one pipeline; `--all` walks `sweep` in order, which is
where experiment 7's combos get appended once results 1–6 are in.

The alternative — `base.yaml` plus per-technique files merged at load — is a
platform pattern that does not pay for itself at seven experiments. It needs a
recursive merge (a shallow `dict.update()` silently drops sibling keys: a
technique file adding one reranker would replace the whole `rerankers:` map), a
file-to-experiment mapping, a rule for whether lists replace or append, and a
`show` command to print the resolved result — because with composition, the
config you ran is no longer a file anyone can read. At ~100 lines total, one file
skips all of it and stays directly inspectable, which matters most in the phase
where pipelines are being added and compared.

*If config composition becomes a goal in itself,* the way to learn it is Hydra or
OmegaConf on top of a working baseline, not a hand-rolled merge — and this
layout is what Hydra's config groups expect, so that stays open. Note that the
config-driven point is already made substantively: seven techniques switch by
config with no code change between them.

**One run, one JSON file** — no tracking system, no manifest abstraction:

```json
{
  "experiment": "hyde_dense",
  "dataset": "nfcorpus",
  "timestamp": "2026-09-02T10:14:00Z",
  "config": { "…the resolved pipeline block, verbatim…" },
  "metrics": {"ndcg@10": 0.31, "recall@100": 0.24, "mrr@10": 0.51},
  "per_query": [
    {
      "query_id": "PLAIN-1",
      "ndcg@10": 0.42,
      "latency_ms": 310,
      "retrieved": ["MED-2438", "MED-1058", "MED-3092"]
    }
  ],
  "llm_calls": 323
}
```

`per_query` is kept because "where did HyDE actually help?" cannot be recovered
from a mean, and re-deriving it costs another 323 LLM calls. `retrieved` holds
the top-k `doc_id`s in rank order — without them a bad score says only *that* a
query failed, not what came back instead, which is the difference between a
number and a diagnosis. Comparing two runs' `retrieved` lists is also how a
combination experiment gets explained: whether reranking promoted a document RRF
had already found, or the fusion surfaced something dense retrieval missed
entirely. At 323 queries × 10 ids that is a few hundred KB per run.

**`document` never reaches disk.** Rows carry `doc_id` and `score` only (§3.1) —
persisting `Document` would put a copy of the corpus in every result file.
`report.py` globs this directory; if the project ever outgrows that, flat JSON is
exactly what an MLflow or W&B import expects.

*Deferred:* a `git_commit` field. Config alone does not pin behaviour — two runs
with identical configs are not comparable if the retriever code changed between
them — so this is the first field to add once the results start being compared
across days rather than within a sitting.

**`llm_calls` counts calls actually made, not cache hits.** A cached re-run of
HyDE would otherwise report zero cost, and that column exists precisely to show
when a technique buying +0.01 nDCG for 323 LLM calls is a bad trade for the AWS
agent.

**The LLM cache is three functions, built in phase 5** — when query expansion
first exists, not before. One JSON file at `.cache/query_expansion.json`, keys
flat:

```json
{"a3f2…": "The relationship between statin use and…",
 "b71c…": ["does statin cause cancer", "statin breast cancer risk"]}
```

One key, one value, whatever shape the call returned. Not nested by technique:
the prompt is already part of `_key(model, prompt, query)`, so HyDE and
multi-query cannot collide, and a `{"hyde": …, "queries": […]}` shape would imply
one hash serves both.

Whole-file load and save is correct at 323 queries — well under a megabyte — but
**save after each call, not at the end of a run**: a sweep that dies on query 200
should keep the first 199, which is the entire point. Without this cache, every
re-run of experiments 3, 4 and any HyDE-containing combo in 7 pays 323 LLM calls
again, and those get re-run more than once while tuning.

**Only NFCorpus, and loading stays in one file.** `nfcorpus.py` holds the
download and the BEIR-format parse (`corpus.jsonl`, `queries.jsonl`,
`qrels/*.tsv`) together. Splitting a generic `loader.py` from a dataset-specific
`nfcorpus.py` now would mean *guessing* which parts are format-generic; a second
dataset shows you where the seam actually is. Extract `common.py` when SciFact
arrives — BEIR sets vary enough (MS MARCO's size, TREC-COVID's qrels shape) that
the abstraction guessed today would likely be the wrong one, which is the scope
creep §7 names as a risk.

**`types.py` stays separate,** because `Doc`/`Query`/`Qrels` are not loading code
— they are the vocabulary `evaluator.py`, `report.py` and the `rag_core` metrics
all speak, and they are the contract that makes a second dataset cheap later. The
separation also keeps the metric tests honest: nDCG is checked against
hand-computed rankings, and those tests should construct a `Qrels` directly
without importing a module that can hit the network.

---

## 5. Planned experiments

| # | Experiment | Question |
|---|---|---|
| 1 | `bm25` | Lexical floor — and the harness validation gate (~0.32) |
| 2 | `dense` | Does semantic search beat lexical on this vocabulary gap? |
| 3 | `hyde_dense` | Does a hypothetical answer close the query/doc distribution gap? |
| 4 | `multiquery_rrf` | Does query diversity + rank fusion beat any single query? |
| 5 | `bi_encoder_rerank` | How much is just "a second, stronger model looked at it"? |
| 6 | `cross_encoder_rerank` | How much more does true cross-attention buy? |
| 7 | 2–3 combos | Stack only what won in 1–6 |

Experiment 7 is deliberately unspecified until 1–6 are in.

The results table reports nDCG@10 and Δ vs BM25, plus recall@100 (did the first
stage even find the documents?), latency and LLM-call count — a technique that
buys 0.01 nDCG for 323 LLM calls is a negative result for the AWS agent, and the
table should say so.

---

## 6. Build order

**A complete pipeline first, then one addition at a time.** Phase 1 goes corpus →
retriever → metric end-to-end with the least abstraction that works; every later
phase adds a single capability to something already producing numbers. No phase
builds machinery that nothing yet consumes.

| Phase | Deliverable |
|---|---|
| 0 | Package scaffold, workspace member |
| 1 | **End-to-end**: load NFCorpus → `SearchResult` + `Retriever` protocol → BM25 → nDCG@10 → **experiment 1** |
| 2 | `embeddings:` resource map, `DenseRetriever`, index cache → experiment 2 |
| 3 | Extract the evaluation runner into `rag_core.evals`; results table |
| 4 | `RRFRetriever`, `RerankingRetriever` + `rerankers:` → experiments 5, 6, hybrid |
| 5 | `llms:`, HyDE, MultiQuery, `llm_cache.py` → experiments 3, 4 |
| 6 | Combos, findings, `results.md` |
| 7 | `RagCore._retriever()` cutover — wrap Pinecone behind the protocol |
| 8 | Fold the winner into the AWS agent's config |

**Phase 1 is the gate.** BM25 landing near the published 0.32 validates dataset
parsing, qrels handling and the nDCG implementation against an external number,
all at once. Nothing later is trustworthy until it does — which is why it comes
first rather than after three phases of unverified groundwork.

Three orderings are deliberate:

- **Pinecone last.** The `_vectorstore()` → `_retriever()` cutover is the only
  change that can break the shipping AWS agent. By phase 7 the protocol has
  carried six retrievers, so the migration is onto something proven rather than
  onto a design still being revised — and no benchmark work sits behind it.
- **The evaluation runner is extracted at two retrievers, not one.** Writing it
  in phase 1 would mean guessing the shape from a single caller. BM25 and dense
  together show which parts are actually generic, which is the same reason
  `loader.py` and `indexes:` are not split up front (§4, §3.2).
- **Resources arrive with their consumers.** `embeddings:` in phase 2 because
  `DenseRetriever` needs it, `rerankers:` in phase 4, `llms:` in phase 5 —
  rather than all three maps built before anything resolves a name.

Composition (phase 4) precedes query expansion (phase 5) even though experiments
5–6 outrank 3–4 in §5: reranking and fusion need no LLM, so three more data
points land before any tokens are spent. Experiment numbering in §5 is unchanged;
only the build order differs.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| nDCG subtly wrong → every number invalid | Phase 1 external gate + hand-computed unit tests |
| `_vectorstore()` → `_retriever()` breaks the agent | Phase 7, last, after the protocol has carried six retrievers; full test suite green |
| Config grows into a DSL | Children nest inline — no cross-references, no expressions, no conditionals; recursion on `type` is the only mechanism |
| A wrong `top_k`/`candidate_k` interaction silently degrades results | Parent-requests-depth rule (§3.2), asserted in a unit test over a nested config |
| HyDE LLM cost across re-runs | Disk-cached by (query, model, prompt); `--limit N` smoke mode |
| Scope creep to a general BEIR harness | One dataset; the protocol generalizes, the package does not |

---

## 8. Open questions

1. **Dependencies** (CLAUDE.md requires approval): `rank_bm25`, `numpy`.
   Deliberately *not* `beir` (drags torch/faiss/elasticsearch) or `pytrec_eval`
   (C extension; ~60 lines of our own is more instructive).
2. **Dense baseline embedding model**: local `nomic-embed-text`, or OpenAI
   `text-embedding-3-small` (~$0.01 for the corpus, comparable to published
   baselines)? Suggest OpenAI for the headline, Ollama as the committed default.
3. **Migration shape for §3.1–3.3**: two blocks change — singular
   `embeddings:`/`llm:` become named maps, and flat `retriever:` + `vectorstore:`
   become one `retrieval.pipelines` entry. Clean break with the AWS `config.yml`
   updated in the same change, or a compat shim that reads the old shape? A shim
   is roughly "wrap the singular block as `{default: ...}`" — cheap for §3.3,
   much less so for §3.2.
4. **Do DPR-style asymmetric encoders make phase 2, or wait?** They are the only
   thing forcing the `{query, passage}` shape, and none of the seven planned
   experiments needs them. Suggest designing the shape now, implementing when an
   experiment actually calls for it.
5. **Does `dense` reuse `rag_core` chunking?** Suggest no — NFCorpus qrels are
   document-level, so chunking would need an arbitrary chunk→doc aggregation and
   break comparability with published numbers.

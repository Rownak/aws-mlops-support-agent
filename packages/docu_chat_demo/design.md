# design — `docu_chat_demo`

Streamlit chat UI over documents uploaded at runtime. Fourth workspace member,
alongside `rag_core`, `aws_mlops_support_agent` and `rag_bench_eval`. Deployed
to Streamlit Community Cloud (free tier) as a second app from this repo.

Import name `docu_chat_demo`, distribution `docu-chat-demo`.

> **Build scope.** This document is the full design. The actual 60-minute demo
> build is a subset — see [`sprints/`](sprints/README.md), which lists exactly
> what is cut (separate modules, tests, config file, deploy, most of §10) and
> what is kept. Read the sprints to build; read this to know what was traded
> away and what to restore before a public deploy.

---

## 1. Goals, scope, non-goals

### Goals
1. Upload PDF / TXT / DOCX through the Streamlit UI.
2. Ingest → chunk → embed → index at runtime, per session.
3. Chat once ingestion completes.
4. Answers grounded **only** in the uploaded documents.
5. Citations with source excerpts under every answer.
6. A clear "not in your documents" reply when the corpus does not cover it.
7. Fits the Community Cloud free tier (~1 GB RAM) alongside the existing demo.

### Scope and assumptions
One user, one browser session, a handful of documents (limits in §8). Everything
lives in memory for the life of that session.

- Embedding + LLM provider configured by env vars, as the AWS agent does it
  (`OPENAI_API_KEY`, or an Ollama `base_url` locally).
- Community Cloud runs one stateful long-lived process; `st.session_state` is
  tied to the browser session.
- The deployed app is public and unauthenticated; the OpenAI hard spend cap is
  the real protection (`deploy/streamlit_cloud_setup.md` step 1).

### Non-goals
No Pinecone or persistent index, no FAISS, no markitdown (§5 states what is used
instead). No auth, no cross-session persistence or document library, no OCR of
scanned PDFs, no agentic tool use or escalation, no eval harness.

---

## 2. User workflow

1. Sidebar shows `st.file_uploader` with accepted types and limits; chat input
   disabled.
2. User selects 1..N files — Streamlit holds them in memory as `UploadedFile`
   (`.name`, `.size`, `.type`, `.getvalue()`).
3. User clicks **Index documents**. An `st.status` panel steps through
   *validate → extract → chunk → embed → index* with a running chunk count.
4. On success the panel collapses to "Indexed 3 documents, 214 chunks"; chat
   input enables.
5. Question → retrieve → grounded answer + numbered citations, each an expander
   with filename and the excerpt the model saw.
6. Uncovered question → refusal message plus a note listing what *is* indexed.
7. **Clear & re-upload** button resets session state back to step 1.

---

## 3. Architecture

```
┌──────────────────────── docu_chat_demo ─────────────────────────┐
│  ui/app.py        Streamlit page: uploader, chat loop, render    │
│  session.py       per-session state (§6)                         │
│  ingest.py        uploaded bytes -> chunks -> in-memory index    │
│  loader.py        PDF/DOCX/TXT -> text  (pypdf, python-docx)     │
│  engine.py        retrieve + generate over the session index     │
│  security.py      file validation (§10)                          │
│  config.py        config.yaml + env + demo limits                │
└───────────────────────────────┬─────────────────────────────────┘
                                │ imports only
┌───────────────────────────────▼─────────────────────────────────┐
│  rag_core:  build_chunks/splitter_from_config · hashing          │
│             get_embedding · get_llm · DenseRetriever             │
│             assess_confidence · AnswerGenerator · Answer         │
└──────────────────────────────────────────────────────────────────┘
```

`rag_core` owns every RAG mechanic and stays untouched — this package adds
nothing to it. `docu_chat_demo` owns the Streamlit UI, session lifecycle, upload
safety, and text extraction. No retrieval, chunking, or generation logic is
reimplemented here.

---

## 4. Data flow

### Ingestion
```
st.file_uploader -> UploadedFile(name, size, type, getvalue())
  → security.validate(name, size, header_bytes)   reject / accept  (§10)
  → loader.extract(bytes, ext)                    -> text
  → sha256(bytes)                                 -> file_hash (rag_core.processing.hashing)
  → build_chunks(text, ..., splitter)             -> list[Document] with provenance
  → SessionIndex.build(chunks, embeddings)        embeds + holds vectors in memory
```

`build_chunks` stamps each chunk with `source`, `file_name`, `file_type`,
`file_hash`, `chunk_index`, `total_chunks` — the provenance the citation UI
needs, so we add no metadata of our own.

### Query
```
question
  → SessionIndex.search(q, k)           -> [(Document, normalized_score)]
  → assess_confidence(scored, min_top_score)
  → AnswerGenerator.generate(q, docs)   -> Answer(text, citations, refused, partial)
  → ui.render(answer, confidence)       -> message + citation expanders
```

Confidence is assessed but **does not** short-circuit generation; it is shown as
a low-confidence hint, and the generator's `REFUSAL_SENTINEL` decides refusals.
If retrieval returns nothing, skip the LLM call.

---

## 5. Integration with `rag_core`

Used as-is. **No changes to `rag_core` are proposed.**

| Need | Reused from `rag_core` |
|---|---|
| Chunking + provenance metadata | `processing.chunking.build_chunks`, `splitter_from_config` |
| File hash | `processing.hashing` |
| Embeddings | `embeddings.factory.get_embedding` |
| LLM | `llm.factory.get_llm` |
| Vector search | `retriever.dense.DenseRetriever` |
| Result shape | `retriever.base.SearchResult` |
| Confidence verdict | `retriever.confidence.assess_confidence` |
| Grounded generation, citations, refusal | `generation.generator.AnswerGenerator` |
| Answer / Citation objects | `generation.answer.Answer`, `Citation` |
| Config schema + env precedence | `config.load_config` → `RagConfig` |
| Logging setup | `utils.logging` |

We compose these parts directly rather than going through the `RagCore` facade,
whose constructor connects to Pinecone. Two consequences to handle:

- **Metadata gap.** `DenseRetriever` builds `SearchResult.document` with only
  `{"doc_id": ...}`, dropping what `build_chunks` produced. `SessionIndex` keeps
  its own `doc_id -> Document` map and re-attaches the full chunk before results
  reach the generator.
- **Score scale.** `DenseRetriever` returns raw cosine in [-1,1];
  `assess_confidence` expects normalized [0,1]. `SessionIndex.search` maps
  `(raw + 1) / 2` before confidence is assessed, so `min_top_score` keeps its
  usual meaning.

**Text extraction is the one seam not reused.** `MarkItDownLoader` is skipped
because `markitdown` pulls `magika` → `onnxruntime` (~47 MB) against a ~1 GB
tier; `pypdf` + `python-docx` cost ~0.7 MB. `loader.py` is ~60 lines —
`pypdf.PdfReader` page text, `python-docx` paragraphs + table cells, UTF-8
decode for TXT — returning plain text into `build_chunks` exactly as
`MarkItDownLoader` would. Accepted loss: markitdown's richer table extraction;
noted in the README.

---

## 6. Session state

Held in `st.session_state["docuchat"]`.

```python
@dataclass
class SessionState:
    status: Literal["awaiting_upload", "ingesting", "ready", "failed"]
    index: SessionIndex | None
    documents: list[DocInfo]        # name, size, chunk count, file_hash
    messages: list[ChatTurn]        # replayed on rerun
    rejected: list[tuple[str, str]] # (filename, reason)
```

```python
class SessionIndex:
    _chunks: dict[str, Document]    # doc_id -> chunk with full metadata
    _retriever: DenseRetriever      # built over {doc_id: page_content}

    def build(chunks: list[Document], embeddings) -> None
    def search(query: str, k: int) -> list[tuple[Document, float]]   # normalized
    def close() -> None
```

Rules:
- **Never `@st.cache_resource` for the index.** That cache is process-wide and
  shared across all sessions; caching a user's documents there leaks uploads
  between visitors. Use it only for the stateless embedding and LLM clients.
- Streamlit re-executes the script on every interaction, so the index lives in
  `st.session_state`, never a module global, and is never rebuilt at script top.
  `messages` is replayed each rerun.
- Chat input is `disabled=(status != "ready")` — the gate for goal 3.
- Chat history is displayed but **not** fed into retrieval; each question is
  answered independently.

---

## 7. Document and index lifecycle

**Nothing is written to disk at any point.** `st.file_uploader` returns
in-memory bytes; `pypdf` and `python-docx` both accept `io.BytesIO`. No temp
dirs, no cleanup path, no traversal risk from filenames, no files left by a
crash.

| Stage | What happens |
|---|---|
| upload | bytes held by Streamlit in session memory |
| validate | header bytes inspected in memory (§10) |
| extract | `io.BytesIO` → text; bytes released immediately after |
| index | chunk text + vectors in `SessionIndex` |
| chat | reads only from `SessionIndex` |
| clear / re-upload | `SessionIndex.close()`, state reset to `awaiting_upload` |
| session end | Streamlit discards `st.session_state`; index garbage-collected |

Only chunk text and vectors persist for the session. `data/` is untouched.

---

## 8. Configuration, logging, error handling

### Configuration
`config.yaml` in the package, committed, non-secret — same schema and
**env > yaml > default** precedence as every other package.

```yaml
embeddings:
  provider: openai
  model: text-embedding-3-small

llm:
  provider: openai
  model: gpt-4o-mini

splitter:
  chunk_size: 800
  chunk_overlap: 100
  strategy: markdown

retriever:
  top_k: 5
  min_top_score: 0.675      # normalized [0,1]; null disables the check

generation:
  max_context_chars: 12000
  system_prompt: null       # null -> rag_core's DEFAULT_SYSTEM_PROMPT (§9)
```

`vectorstore` and `sources` are omitted — this demo builds neither.

> **Confirmed:** `load_config` on a config that omits `vectorstore` still
> raises `Missing required environment variables: ... PINECONE_API_KEY
> (required by vectorstore.provider: pinecone ...)`, because
> `RagConfig._validate()` runs every block's `missing_secrets()`
> unconditionally and `VectorStoreConfig` defaults to `provider: pinecone`.
>
> Two fixes, neither touching `rag_core`: set a dummy `vectorstore.host` (any
> string — nothing connects, verified), or skip `load_config` and use plain
> constants. **The sprint build uses constants**; this section applies only if
> a config file is reintroduced later.

Demo-only limits are plain constants in `docu_chat_demo.config`:
`MAX_FILES = 5`, `MAX_FILE_MB = 20`, `MAX_TOTAL_MB = 50`,
`ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}`,
`MAX_QUESTIONS_PER_SESSION = 5` (mirrors the AWS demo's courtesy cap; refresh
resets it, the spend cap is the real control).

### Logging
`rag_core.utils.logging` at startup. One INFO line per stage per file
(`extract`, `chunk` with count, `embed`, `index`), one per query with
`top_score` and `refused`. **Never logged:** file contents, chunk text, the
user's question verbatim, or any key value. Filenames only.

### Error handling
Every failure renders as actionable UI, never a traceback.

| Failure | Behavior |
|---|---|
| Rejected file type/size | Listed in `rejected` with the rule; other files still ingest |
| Empty extracted text (e.g. scanned PDF) | Skipped — "no extractable text; scanned PDFs are not supported" |
| **All** files fail | `status = "failed"`, uploader stays open with reasons |
| Embedding/LLM provider error | `st.error` "model provider unavailable"; status stays `ready` so the user can retry |
| Unexpected exception | Traceback logged server-side; generic `st.error` client-side |

Partial success is the rule: three good files and one bad gives a working chat
over three documents plus a note about the fourth.

---

## 9. Grounding, citations, unsupported questions

All three come from `rag_core`; no new logic.

**Grounding.** `DEFAULT_SYSTEM_PROMPT` already forbids outside knowledge,
mandates `[n]` citations, and defines `INSUFFICIENT_CONTEXT` as the refusal
token. `prompts.py` holds at most a thin override adding the demo's framing
("the user's uploaded documents"); the rules are not rewritten.

**Citations.** `ui/render.py` renders each `Citation` as an `st.expander`
labelled `[n] filename` containing the excerpt the model saw — the chunk text
itself, not a re-retrieval. `Answer.confidence` is citation coverage, **not**
correctness; shown as a caption worded so it cannot be read as a probability of
being right.

**Unsupported questions**, three paths:
- `answer.refused` → `REFUSAL_MESSAGE` plus a line listing indexed filenames.
  No citations shown.
- `answer.partial` → the answer with citations **and** the partial-answer caveat
  above it.
- Not confident but not refused → the answer with a "weak match" caption. We
  report the verdict; we do not suppress the answer.

---

## 10. Security for uploaded files

The deployed app is public and unauthenticated.

1. **Extension + MIME allowlist.** `.pdf/.txt/.docx` and their expected
   `UploadedFile.type` must both match. `st.file_uploader(type=[...])` is a UI
   filter, not a control — `security.validate` re-checks server-side.
2. **Magic bytes.** `%PDF-`, `PK\x03\x04` for docx, valid UTF-8 for txt. Catches
   the rename trick extension checks miss.
3. **Size and count caps** (§8) enforced before extraction. Set Streamlit's
   `server.maxUploadSize` to match so oversize files are refused at the
   transport layer.
4. **No filesystem, no traversal.** Nothing is written to disk (§7), so the
   filename is never a path component — kept only as a display string in chunk
   metadata, HTML-escaped before rendering.
5. **Zip-bomb guard for DOCX.** Check the sum of `ZipInfo.file_size` against a
   cap before extracting.
6. **No execution.** Files are only parsed by `pypdf`/`python-docx`; nothing is
   executed or shelled out.
7. **Parser risk acknowledged, not eliminated.** Both parsers are third-party;
   size caps and in-memory handling limit blast radius. A hardened deployment
   would sandbox extraction in a subprocess — out of scope for a demo.
8. **Prompt injection.** Uploaded text reaches the LLM as *sources* and may try
   to instruct it. The `[n]`-citation requirement makes injected instructions
   visible, and the demo has no tools to hijack. Residual risk is a misleading
   answer; noted in the README.
9. **Spend.** This demo bills embeddings on every upload *as well as*
   completions. The project-scoped OpenAI key with a hard cap is the defence.
10. **Secrets** only from env vars / Community Cloud Secrets; never logged.

---

## 11. Folder structure

```
packages/docu_chat_demo/
├── pyproject.toml              deps: rag-core, streamlit, pypdf, python-docx
├── config.yaml
├── design.md
├── progress.md
├── README.md
├── src/docu_chat_demo/
│   ├── __init__.py
│   ├── config.py               load_config + demo limits
│   ├── loader.py               PDF/DOCX/TXT -> text
│   ├── security.py             validation (§10)
│   ├── session.py              SessionState, SessionIndex
│   ├── ingest.py               validate -> extract -> chunk -> index
│   ├── engine.py               search + confidence + generate
│   ├── prompts.py
│   └── ui/
│       ├── __init__.py
│       ├── app.py              the Streamlit page (main())
│       ├── upload.py           uploader widget + index button
│       └── render.py           Answer -> chat message + citation expanders
└── tests/
    ├── test_security.py
    ├── test_loader.py
    ├── test_ingest.py
    ├── test_session_index.py
    ├── test_engine.py
    └── fixtures/               tiny sample.pdf / sample.docx / sample.txt

app_docuchat/                   Community Cloud entrypoint (§12) — NOT in packages/
├── streamlit_app.py            sys.path shim -> docu_chat_demo.ui.app:main
├── requirements.txt            exported pins; must sit beside the entrypoint
└── .streamlit/config.toml      server.maxUploadSize
```

`ui/app.py` holds widgets and the chat loop only, delegating to the modules
beside it.

---

## 12. Deployment (Streamlit Community Cloud)

A **second, independent app** from this repo; the rules in
`deploy/streamlit_cloud_setup.md` apply again rather than being inherited.

**Entrypoint rule.** Community Cloud uses the *first* dependency file it finds,
searching the entrypoint's own directory before the repo root, in the order
`uv.lock, Pipfile, environment.yml, requirements.txt, pyproject.toml`. The root
`uv.lock` outranks everything and drags in the whole workspace, including
`rag_core`'s `[tool.uv.sources]` `extra` key, which Community Cloud's older `uv`
rejects. So the entrypoint and its `requirements.txt` must sit together in
`app_docuchat/`. Do not move either out on its own; do not point **Main file
path** at anything under `packages/`.

The shim prepends `packages/rag_core/src` and `packages/docu_chat_demo/src` to
`sys.path` (workspace packages are never `pip install`ed by the host) and calls
`docu_chat_demo.ui.app:main`.

```bash
uv export --frozen --no-dev --no-emit-workspace --no-annotate --no-header \
  --package docu-chat-demo --no-hashes -o app_docuchat/requirements.txt
```

**Settings:** Main file path `app_docuchat/streamlit_app.py`, Python 3.13,
Secrets as TOML — `OPENAI_API_KEY` only, no Pinecone key (§8).

**Budget.** ~1 GB RAM. Additions over the existing demo's baseline are `pypdf`
(0.4 MB) + `python-docx` (0.3 MB); numpy is already present. Session vectors are
~1.2 MB at the §8 limits.

---

## 13. Testing

Runnable in isolation, no Streamlit process required. `ui/app.py` is the only
module not directly unit-tested, which is why it holds no logic.

| Test | What it proves |
|---|---|
| `test_security.py` | Each rejection fires: bad extension, MIME/extension mismatch, wrong magic bytes, oversize, too many files, docx zip-bomb. Includes a `.txt` renamed to `.pdf`. |
| `test_loader.py` | Each of PDF/TXT/DOCX extracts expected text; a text-less PDF returns empty (not a crash); a corrupt file raises a handled error. |
| `test_ingest.py` | Chunks carry correct `source`/`file_name`/`file_hash`; one bad file among good ones is skipped and reported without killing the batch. |
| `test_session_index.py` | With a **stub embedding** (deterministic, no network): expected chunk ranks first, full metadata re-attached, scores in [0,1], `close()` drops state. |
| `test_engine.py` | With a **stub LLM**: supported question yields citations; irrelevant sources yield `refused=True` and no citations; an empty index never calls the LLM. |

Stubs live in `tests/`; no test helpers are added to `rag_core`. Whole workspace
suite green (`uv run pytest` from the root) before any commit message is drafted.

### Acceptance criteria
1. Upload one PDF, one TXT, one DOCX → all ingest; chat enables.
2. Answerable question → correct answer, ≥1 citation, excerpt traceable to the
   right file.
3. Unrelated question ("who won the 2019 World Cup") → refusal, no invented
   answer, no citations.
4. Asking before indexing finishes → input disabled, no LLM call.
5. `.exe`, oversize, or renamed `.txt→.pdf` → rejected with a reason; valid
   files in the same batch still work.
6. Clear & re-upload → old index released; no answer cites a previous document.
7. Scanned PDF with no text layer → skipped with an explanation, not a crash.
8. Two browser sessions with different documents never see each other's content.
9. `uv run pytest` green across the workspace.
10. Deployed app boots on Community Cloud and passes 1–8 against the public URL.

---

## 14. Implementation plan

| Phase | Work | Done when |
|---|---|---|
| **1. Skeleton** | Package, `pyproject.toml` (`rag-core`, `streamlit`, `pypdf`, `python-docx`), workspace registration in root `pyproject.toml` (members, `tool.uv.sources`, `testpaths`), `config.yaml`, config loading. Settle the Pinecone-key check in §8. | `uv sync` installs; config loads with no Pinecone key present |
| **2. Loader + security** | `loader.py`, `security.py`, fixtures, their tests. No UI. | Tests pass; a script turns three fixtures into text |
| **3. Ingest** | `ingest.py`: validate → extract → hash → `build_chunks`, progress callback, per-file error capture. | `test_ingest.py` passes |
| **4. Session index** | `session.py`: `SessionIndex` over `DenseRetriever`, metadata re-attachment, score normalization, `close()`. | `test_session_index.py` passes |
| **5. Engine** | `engine.py`: search → confidence → `AnswerGenerator`. `prompts.py`. | `test_engine.py` passes; a CLI script answers over a fixture doc |
| **6. Streamlit UI** | `ui/app.py`, `ui/upload.py`, `ui/render.py`. Uploader, `st.status` progress, chat loop, citation expanders, `disabled` gate, clear & re-upload. | Runs locally end to end; criteria 1–8 pass by hand |
| **7. Deploy** | `app_docuchat/` shim, exported `requirements.txt`, `.streamlit/config.toml`, README, notes appended to `deploy/streamlit_cloud_setup.md`. | Criterion 10 passes against the public URL |

Phases 2–5 are pure library code with no Streamlit import, which keeps phase 6 a
thin wiring job.

### Stretch (not in the plan)
Follow-up query rewriting from chat history, hybrid BM25+dense retrieval
(`rag_core` has `BM25Retriever` and `RRFRetriever`), FAISS or a persistent store
if the corpus outgrows RAM, markitdown-quality table extraction if the RAM
budget lifts.

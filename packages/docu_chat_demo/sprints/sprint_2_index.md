# Sprint 2: Ingest and Index

**Goal:** Extracted text becomes chunks with provenance metadata, embedded into
an in-memory index held in session state. No chat yet.

**Time:** 15 minutes

---

## Phase 2.1 - Chunking

- [ ] At the top of `core.py`, add the constants (no config file):

```python
EMBED = {"provider": "openai", "model": "text-embedding-3-small"}
LLM = {"provider": "openai", "model": "gpt-4o-mini"}
CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 800, 100, 5
```

- [ ] Add `build_docs(files: list[tuple[str, str]]) -> list[Document]`
- [ ] Use `rag_core.processing.chunking.get_markdown_splitter(CHUNK_SIZE, CHUNK_OVERLAP)`
- [ ] Per file: `sha256_file(data)` from `rag_core.processing.hashing` for the
      hash, then `build_chunks(text, name, name, ext, file_hash, splitter)`
- [ ] Return the concatenated chunk list

> `build_chunks` already stamps `source`, `file_name`, `file_type`,
> `file_hash`, `chunk_index`, `total_chunks` — that is the citation provenance,
> so add no metadata of your own.

## Phase 2.2 - In-Memory Index

- [ ] Add `build_index(docs) -> tuple[DenseRetriever, dict[str, Document]]`
- [ ] Build `corpus = {d.metadata["chunk_id"]: d.page_content for d in docs}`
- [ ] Keep `by_id = {chunk_id: doc}` — **required**, because
      `DenseRetriever` returns `SearchResult.document` with only
      `{"doc_id": ...}` and drops the real metadata
- [ ] `DenseRetriever(corpus, get_embedding(EMBED), top_k=TOP_K)`
- [ ] Cache the embedding client with `@st.cache_resource` in `app.py`

> **Never `@st.cache_resource` the index or the corpus.** That cache is
> process-wide; caching a user's documents there serves them to other browser
> sessions. Cache only the stateless embedding/LLM clients.

## Phase 2.3 - Session Wiring

- [ ] In `app.py`, add an **Index documents** button
- [ ] On click, inside `with st.status("Indexing..."):` run extract → chunk →
      embed
- [ ] Store `st.session_state["index"]`, `["by_id"]`, `["docs"]` (filenames)
- [ ] Show "Indexed N documents, M chunks" on completion

---

## Expected output

Clicking **Index documents** shows a progress panel, then a success line with
correct document and chunk counts. Re-running the script (any widget
interaction) does not rebuild the index.

## Verify

```bash
uv run streamlit run packages/docu_chat_demo/src/docu_chat_demo/app.py
```

Upload 3 files → **Index documents** → "Indexed 3 documents, N chunks" with
N > 0. Interact with any widget; the count stays and no re-embedding happens
(no new OpenAI latency).

Non-UI sanity check that retrieval returns the right chunk:

```bash
uv run python -c "
import sys; sys.path.insert(0,'packages/docu_chat_demo/src')
from docu_chat_demo.core import build_docs, build_index
docs = build_docs([('a.txt','The mitochondrion is the powerhouse of the cell.'),
                   ('b.txt','Invoices are due within 30 days of receipt.')])
r, by_id = build_index(docs)
top = r.search('When are invoices due?', 1)[0]
print(by_id[top.doc_id].metadata['file_name'], '|', round(top.score, 3))
"
```

Expect `b.txt` and a positive score.

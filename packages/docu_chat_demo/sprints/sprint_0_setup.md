# Sprint 0: Project Setup

**Goal:** `docu_chat_demo` exists as a workspace package, dependencies resolve,
and a placeholder Streamlit page opens in the browser.

**Time:** 10 minutes

---

## Phase 0.1 - Dependency Setup

- [ ] Create `packages/docu_chat_demo/pyproject.toml` (name `docu-chat-demo`,
      `requires-python = ">=3.13, <4.0"`, hatchling build, wheel packages
      `src/docu_chat_demo`)
- [ ] Dependencies — exactly four: `rag-core`, `streamlit>=1.58`, `pypdf`,
      `python-docx`
- [ ] Add `rag-core = { workspace = true }` under `[tool.uv.sources]`
- [ ] Register in the **root** `pyproject.toml`: add `"docu-chat-demo"` to
      `dependencies` and to `[tool.uv.sources]` as `{ workspace = true }`
- [ ] `uv sync`

> No `--extra` needed. Do **not** add `rag-core[ingest]` — the PDF/DOCX parsing
> here uses `pypdf`/`python-docx` directly, and `ingest` pulls ~47 MB
> (markitdown → magika → onnxruntime).

## Phase 0.2 - Package Structure

- [ ] Create the minimal folder structure — **two Python files, no subpackages**:

```
packages/docu_chat_demo/
├── pyproject.toml
└── src/docu_chat_demo/
    ├── __init__.py       (empty)
    ├── core.py           (load + chunk + index + answer)
    └── app.py            (Streamlit page)
```

- [ ] Add the Streamlit entry point — `app.py` with a title and
      `st.file_uploader`, nothing wired yet:

```python
import streamlit as st

st.set_page_config(page_title="Document Chat", layout="centered")
st.title("Document Chat")
st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
)
```

- [ ] Leave `core.py` empty for now

> **Config decision (saves a file and a real bug).** There is no `config.yaml`
> and we do **not** call `rag_core.config.load_config`. Verified: with a config
> that omits `vectorstore`, `load_config` still raises
> `Missing required environment variables: ... PINECONE_API_KEY (required by
> vectorstore.provider: pinecone ...)` — because `RagConfig._validate()` runs
> every block's `missing_secrets()` unconditionally. Put plain constants at the
> top of `core.py` instead (Sprint 2). `OPENAI_API_KEY` is read by the OpenAI
> client itself from `.env`.

---

## Expected output

`uv sync` completes, and a browser page titled **Document Chat** shows an
upload widget that accepts pdf/txt/docx.

## Verify

```bash
uv run python -c "import pypdf, docx, streamlit; print('deps ok')"
uv run streamlit run packages/docu_chat_demo/src/docu_chat_demo/app.py
```

Browser opens at `localhost:8501`, title and uploader visible. Ctrl-C to stop.

Also confirm nothing broke:

```bash
uv run pytest -q
```

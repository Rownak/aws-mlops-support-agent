# Sprints — `docu_chat_demo` (60-minute build)

Five sprints, 60 minutes total. Each is a separate file; do them in order.

| Sprint | Goal | Time |
|---|---|---|
| [0](sprint_0_setup.md) | Package + deps + Streamlit "hello" that runs | 10 min |
| [1](sprint_1_upload_parse.md) | Upload PDF/TXT/DOCX → extracted text on screen | 10 min |
| [2](sprint_2_index.md) | Text → chunks → in-memory vector index | 15 min |
| [3](sprint_3_chat.md) | Grounded chat + citations | 15 min |
| [4](sprint_4_e2e.md) | End-to-end test + README | 10 min |

## What was cut from `design.md` to fit the hour

The design doc is the full picture; this plan is the demo subset. Dropped:

- **Separate `security.py`, `loader.py`, `engine.py`, `ingest.py`, `session.py`
  modules** → collapsed into **two** modules: `core.py` (load + chunk + index +
  answer) and `app.py` (Streamlit).
- **The `tests/` suite** (5 test files) → one manual end-to-end check in
  Sprint 4.
- **`config.yaml` + `rag_core.config.load_config`** → constants at the top of
  `core.py`. Avoids the Pinecone-key validation problem entirely (see Sprint 0
  note) and saves a file.
- **Community Cloud deploy** (`app_docuchat/` shim, pinned requirements) →
  local `streamlit run` only. Deploy is a follow-up, not part of the hour.
- **Zip-bomb guard, MIME allowlist, magic-byte checks** → extension check only.
  Acceptable for a local demo; **re-add from design.md §10 before any public
  deploy.**
- **Clear & re-upload, per-session question cap, partial-answer caveat
  rendering** → skipped.

## What is deliberately kept

These are cheap and are what the demo is judged on:

- Answers grounded only in uploaded docs (`rag_core`'s prompt does this).
- Citations with source excerpts.
- A real refusal when the documents do not cover the question.
- `@st.cache_resource` used **only** for the embedding/LLM clients, never the
  index — caching the index there leaks documents between browser sessions.

## Prerequisites

- `OPENAI_API_KEY` in the repo's `.env` (already present for the AWS demo).
- Three small test files ready: one `.pdf`, one `.txt`, one `.docx`.

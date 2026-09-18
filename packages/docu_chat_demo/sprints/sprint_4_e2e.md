# Sprint 4: End-to-End Check

**Goal:** Prove the whole flow works on real files, confirm nothing else in the
workspace broke, and leave a README so the demo can be run again cold.

**Time:** 10 minutes

---

## Phase 4.1 - Manual End-to-End Run

Run the app and walk the full path with three real files:

```bash
uv run streamlit run packages/docu_chat_demo/src/docu_chat_demo/app.py
```

- [ ] Chat input is disabled before indexing
- [ ] Upload one `.pdf`, one `.txt`, one `.docx` → all three parse
- [ ] Index → "Indexed 3 documents, N chunks"
- [ ] Question answerable from the PDF → cited answer, excerpt matches the file
- [ ] Question answerable from the DOCX → cited answer from *that* file
- [ ] Off-topic question → refusal, no citations
- [ ] A renamed/corrupt file among good ones → warning only, others still index

## Phase 4.2 - Regression and Docs

- [ ] `uv run pytest -q` — the existing suite must stay green
- [ ] `uv run ruff check packages/docu_chat_demo`
- [ ] Write a short `packages/docu_chat_demo/README.md`: one-line purpose, the
      `streamlit run` command, the `OPENAI_API_KEY` requirement, and a
      **Limitations** section
- [ ] In Limitations, state plainly: extension check only (no MIME/magic-byte
      validation), no OCR for scanned PDFs, index is per-session and lost on
      refresh, local-only — not deployed
- [ ] Add 2-3 lines to a new `packages/docu_chat_demo/progress.md`
- [ ] Stage the new files (`git add`) and draft a commit message — **do not
      commit**

---

## Expected output

All seven checks in 4.1 pass, `pytest` is green, and the package has a README
whose limitations are honest about what was cut.

## Verify

```bash
uv run pytest -q && uv run ruff check packages/docu_chat_demo && git status --short
```

Green tests, clean lint, and only `packages/docu_chat_demo/` files staged.

---

## If time remains

In rough order of value, from `design.md`:

1. **Security hardening** (§10) — MIME + magic-byte checks, size caps, docx
   zip-bomb guard. **Required before any public deploy**, since the deployed
   app would be unauthenticated.
2. **Clear & re-upload button** — reset session state, ~5 lines.
3. **Community Cloud deploy** (§12) — needs `app_docuchat/` with the entrypoint
   and its own `requirements.txt` side by side, or the root `uv.lock` is picked
   up and the build fails on `rag_core`'s `extra` key.
4. **Tests** (§13) — start with the refusal path and the metadata re-attachment.

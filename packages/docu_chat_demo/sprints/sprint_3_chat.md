# Sprint 3: Grounded Chat and Citations

**Goal:** Ask questions and get answers grounded only in the uploaded
documents, each with citation excerpts, and a clear refusal when the documents
do not cover the question.

**Time:** 15 minutes

---

## Phase 3.1 - Answering

- [ ] In `core.py`, add `answer(question, retriever, by_id) -> Answer`
- [ ] `results = retriever.search(question, TOP_K)`
- [ ] Re-attach metadata: `docs = [by_id[r.doc_id] for r in results]` — do
      **not** pass `r.document`, its metadata is stripped
- [ ] If `docs` is empty, return early without calling the LLM
- [ ] `AnswerGenerator(llm=get_llm(LLM)).generate(question, docs)`
- [ ] Cache the LLM client with `@st.cache_resource`

> Grounding and refusal need **no prompt of your own**: `AnswerGenerator`'s
> `DEFAULT_SYSTEM_PROMPT` already forbids outside knowledge, mandates `[n]`
> citations, and emits `INSUFFICIENT_CONTEXT` when the sources fall short —
> which `Answer.refused` exposes as a bool.

## Phase 3.2 - Chat UI

- [ ] Keep `st.session_state["messages"]` as a list of `(role, text)`
- [ ] Replay history each rerun with `st.chat_message`
- [ ] `st.chat_input("Ask about your documents", disabled=not indexed)` — the
      disabled flag is the "wait until ingestion completes" gate
- [ ] On submit: call `answer(...)`, append both turns, render

## Phase 3.3 - Citations and Refusal

- [ ] If `ans.refused` → show the text plus a line listing the indexed
      filenames; render **no** citations
- [ ] Otherwise → show `ans.text`, then one `st.expander` per citation labelled
      `[{c.index}] {c.source}` containing `c.snippet`
- [ ] Add `st.caption` for citation coverage, worded as coverage not accuracy,
      e.g. `f"{ans.confidence:.0%} of sentences carry a citation"`

---

## Expected output

A question the documents answer returns a cited answer with expandable source
excerpts. An unrelated question returns a refusal with no citations. The chat
box is greyed out until indexing has run.

## Verify

```bash
uv run streamlit run packages/docu_chat_demo/src/docu_chat_demo/app.py
```

1. Before indexing → chat input is disabled.
2. Index, then ask something the documents answer → answer with `[1]`-style
   markers and at least one expander whose excerpt really comes from that file.
3. Ask **"Who won the 2019 Cricket World Cup?"** → refusal, no citations, no
   invented answer.

Non-UI check of the refusal path:

```bash
uv run python -c "
import sys; sys.path.insert(0,'packages/docu_chat_demo/src')
from docu_chat_demo.core import build_docs, build_index, answer
docs = build_docs([('policy.txt','Invoices are due within 30 days of receipt.')])
r, by_id = build_index(docs)
a1 = answer('When are invoices due?', r, by_id)
a2 = answer('Who won the 2019 Cricket World Cup?', r, by_id)
print('grounded refused =', a1.refused, '| citations =', len(a1.citations))
print('offtopic  refused =', a2.refused)
"
```

Expect `grounded refused = False` with ≥1 citation, and `offtopic refused = True`.

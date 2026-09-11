# tasks v2.3 — aws_mlops_support_agent deploy

## Phase 1 — Fix ingestion UnicodeDecodeError on non-ASCII markdown docs [DONE]

Root cause (deeper than first thought): MarkItDown's `PlainTextConverter`
guesses a text file's charset by sampling only the first 4KB (via
`charset_normalizer`), then decodes the *entire* file with that one guess.
The CloudFormation doc is pure ASCII in its first 4KB but has a non-ASCII
byte at position 13096 — so the sniffed guess was `ascii`, applied file-wide.
Switching `convert()` -> `convert_stream()` alone does not fix this (verified
by reproducing the failure post-switch); the guess itself is wrong.

Fix, in `rag_core/loaders/markitdown_loader.py`: for text extensions
(`.txt`, `.md`, `.markdown`, `.json`, `.jsonl`), pass `StreamInfo(charset="utf-8",
extension=suffix)` into `convert_stream()`, pinning UTF-8 explicitly instead
of trusting the 4KB-sampled guess. Our AWS docs corpus is UTF-8, so this is
correct, not a workaround.

- [x] Update `MarkItDownLoader.load()` to force `charset="utf-8"` via
      `StreamInfo` for text extensions.
- [x] Add `rag_core` regression test (`test_markitdown_loader.py`) with a
      fixture >4KB of ASCII followed by a non-ASCII char, matching the real
      failure shape (a small-file fixture wouldn't have caught this).
- [x] Re-ran ingestion over all 336 `.md` files in `data/aws_docs/` — 0
      failures, including the original CloudFormation doc.
- [x] `uv run pytest` — 298 passed, 1 pre-existing unrelated failure
      (`test_rag_config_loaded_from_project_config_yml`, caused by local
      uncommitted `config.yml` change, not by this fix).

## Phase 2 — Add flag to save chunks locally during ingestion [DONE]

`aws-agent-ingest` (`ingest.py`) calls `RagCore.sync()` -> `Ingestor.sync()`,
which builds chunks in `_prepare_one()` (`rag_core/ingestion/ingestor.py:105`,
`build_chunks(...)`) and writes them to Pinecone via `_write_chunks()`. For
local testing, chunks should optionally also be dumped to disk alongside
the Pinecone upload.

Seam (decided): `Ingestor` grows an optional `on_chunks_prepared` callback,
called per document in `_write_prepared()` right after chunks are built —
additive, so the default path (no callback) is unchanged and the AWS agent
keeps working untouched.

Dump path (decided): `data/aws_docs/chunks/{source_id}/{doc_stem}/`, e.g.
chunks for `data/aws_docs/codebuild/doc_source/access-tokens.md` go to
`data/aws_docs/chunks/codebuild/access-tokens/`. `source_id`/`doc_stem`
aren't in chunk metadata today — derive from `file_path`
(`data/aws_docs/{source_id}/doc_source/{doc_stem}.md`), since
`AwsDocsGitSource` always lays out clones this way; don't add a new
metadata field for this.

- [x] Added `on_chunks_prepared: Callable[[str, list[Document]], None] | None`
      param to `Ingestor.__init__` (default `None`), invoked in
      `_write_prepared()` right after a successful chunk build (not called
      for skipped/failed files). `RagCore.__init__` forwards it through.
- [x] Added `--save-chunks-local` flag to `aws-agent-ingest` (`ingest.py`);
      when set, `_save_chunks_local()` parses `source_id`/`doc_stem` from
      `file_path` via regex (matching `AwsDocsGitSource`'s
      `{source_id}/doc_source/{doc_stem}.md` layout, with an `_unknown`
      fallback for other shapes) and writes one JSON file per chunk
      (`text` + `metadata`) under
      `data/aws_docs/chunks/{source_id}/{doc_stem}/chunk_0000.json` etc.
- [x] `data/aws_docs/chunks/` covered by the existing blanket `data/` entry
      in `.gitignore` — no change needed.
- [x] Tests: `rag_core/tests/test_ingest.py` covers the callback (not called
      by default; called with file_path+chunks; skipped for skip/error
      records). `aws_mlops_support_agent/tests/test_ingest.py` covers
      `_save_chunks_local`'s path derivation and fallback.
- [x] `uv run pytest` — 303 passed, 1 pre-existing unrelated failure
      (same `test_rag_config_loaded_from_project_config_yml`, local
      `config.yml` edit, not this change).

## Phase 3 — Partial-answer mode instead of hard refusal

Today, when retrieved excerpts don't fully answer the question, the system
prompt (`ANSWER_SYSTEM_PROMPT`) instructs the model to reply with exactly
`REFUSAL_SENTINEL` and nothing else. `generator._is_refusal()` catches this,
and the caller sees a generic `REFUSAL_MESSAGE`, discarding whatever the
excerpts *did* say — even when they're on-topic background a user would
still find useful. Observed via a LangSmith trace: query "explain aws ci/cd
pipeline" retrieved two on-topic (score 0.825) but purely definitional
excerpts (what CodePipeline/CI/CD *are*, not pipeline stages/mechanics); the
model correctly followed instructions and refused, but a partial, cited
summary with a caveat would have served the user better. rag_core changes
stay additive (per CLAUDE.md): a new `PARTIAL_SENTINEL` sits alongside
`REFUSAL_SENTINEL`, `_is_refusal()`'s total-refusal path is untouched, and
`rag_bench_eval`/other `rag_core` consumers are unaffected.

- [x] **3.1 rag_core: add the partial-answer sentinel and field.** Added
  `PARTIAL_SENTINEL = "PARTIAL_ANSWER_CAVEAT"` next to `REFUSAL_SENTINEL` in
  `answer.py`, plus `Answer.partial: bool` (default `False`, independent of
  `refused`, threaded through `to_dict()`/`__repr__()`). In `generator.py`,
  `_finalize()` now runs `_extract_partial_caveat()` (a regex matching a
  trailing `PARTIAL_ANSWER_CAVEAT: <note>` line) after the existing
  `_is_refusal()` check: if found, the line is stripped, citations/confidence
  are computed from the remaining text, `partial=True`/`refused=False` are
  set, and the model's own note is appended back via a `PARTIAL_ANSWER_PREFIX`
  sentence so the user sees a clear caveat. A degenerate case — the model
  emits only the sentinel line with no real answer text — falls back to the
  normal refusal path rather than returning an empty "partial" answer.
  `AnswerGenerator.__init__` also substitutes a `{partial_sentinel}`
  placeholder (alongside the existing `{sentinel}`) so a project's prompt can
  reference it without hardcoding the string. `_is_refusal()` itself is
  unchanged.

- [x] **3.2 aws_mlops_support_agent: update the system prompt.** In
  `prompts.py`, `ANSWER_SYSTEM_PROMPT` keeps `{sentinel}` for excerpts wholly
  unrelated to the question, and now has a middle rule: if the excerpts only
  partially answer or give related background, summarize what they say with
  normal `[1][2]` citations, then end with a line of exactly
  `{partial_sentinel}: <short note on what's missing>`.

- [x] **3.3 Tests.** Added to `rag_core/tests/test_answer.py` (sentinel
  distinctness, `Answer.partial` default/independence from `refused`) and
  `rag_core/tests/test_generator.py` (placeholder substitution, a partial
  answer detected/stripped/re-noted correctly, a bare-sentinel reply treated
  as refusal, and a normal answer confirmed *not* partial) — 6 new tests, all
  existing refusal/normal-answer cases untouched and still passing.
  `uv run pytest`: `rag_core` 225 passed; `aws_mlops_support_agent` 70
  passed, 1 pre-existing unrelated failure
  (`test_rag_config_loaded_from_project_config_yml`, local `config.yml`
  drift from prior phases, not this change).

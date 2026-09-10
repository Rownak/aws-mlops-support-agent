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

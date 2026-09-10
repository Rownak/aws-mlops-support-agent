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
      uncommitted `config.yml` change swapping codebuild for codepipeline,
      not by this fix).

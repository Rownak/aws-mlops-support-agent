"""Regression test for the ascii-decode ingestion failure.

MarkItDown guesses a text file's charset by sampling only the first 4KB via
charset_normalizer, then decodes the *entire* file with that guess. A file
that is pure ASCII in its first 4KB but has a non-ASCII byte later (as in
the CloudFormation doc that surfaced this bug, which failed at byte 13096)
gets an "ascii" guess applied to the whole file, raising UnicodeDecodeError.

MarkItDownLoader.load() now pins `charset="utf-8"` explicitly for text
extensions instead of trusting that sniffed guess.
"""

from rag_core.loaders.markitdown_loader import MarkItDownLoader


def test_load_handles_non_ascii_byte_past_the_4kb_sniff_window(tmp_path):
    md_file = tmp_path / "non_ascii.md"
    # >4KB of pure ASCII, then a UTF-8 multi-byte char (em dash): charset
    # sniffing on the first 4KB alone would guess "ascii" and blow up here.
    padding = "safe ascii text\n" * 400
    md_file.write_text(padding + "Deploy — it's automatic.", encoding="utf-8")

    loader = MarkItDownLoader()
    result = loader.load(md_file)

    assert result["success"], result["error"]
    assert "Deploy" in result["text_content"]

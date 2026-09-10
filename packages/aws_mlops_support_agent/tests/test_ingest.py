"""Tests for the --save-chunks-local local chunk dump in ingest.py."""

import json

from langchain_core.documents import Document

from aws_mlops_support_agent.ingest import _save_chunks_local


def test_save_chunks_local_derives_source_id_and_doc_stem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = str(
        tmp_path / "data" / "aws_docs" / "codebuild" / "doc_source" / "access-tokens.md"
    )
    chunks = [
        Document(page_content="chunk zero", metadata={"chunk_index": 0}),
        Document(page_content="chunk one", metadata={"chunk_index": 1}),
    ]

    _save_chunks_local(file_path, chunks)

    out_dir = tmp_path / "data" / "aws_docs" / "chunks" / "codebuild" / "access-tokens"
    assert (out_dir / "chunk_0000.json").exists()
    assert (out_dir / "chunk_0001.json").exists()

    saved = json.loads((out_dir / "chunk_0000.json").read_text(encoding="utf-8"))
    assert saved["text"] == "chunk zero"
    assert saved["metadata"]["chunk_index"] == 0


def test_save_chunks_local_falls_back_for_unrecognized_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    chunks = [Document(page_content="x", metadata={"chunk_index": 0})]

    _save_chunks_local("some/other/layout/doc.md", chunks)

    out_dir = tmp_path / "data" / "aws_docs" / "chunks" / "_unknown" / "doc"
    assert (out_dir / "chunk_0000.json").exists()

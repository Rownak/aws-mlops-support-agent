from rag_bench_eval.datasets.types import Doc


def test_content_joins_title_and_text():
    doc = Doc(doc_id="1", title="Title", text="Body text")
    assert doc.content == "Title Body text"


def test_content_falls_back_to_text_when_title_empty():
    doc = Doc(doc_id="1", title="", text="Body text")
    assert doc.content == "Body text"

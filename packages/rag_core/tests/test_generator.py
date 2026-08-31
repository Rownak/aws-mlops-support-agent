"""Tests for AnswerGenerator, exercised with a fake LLM (no network)."""

from rag_core.generation.answer import REFUSAL_SENTINEL
from rag_core.generation.generator import AnswerGenerator


class _FakeDoc:
    def __init__(self, text, source="doc.md"):
        self.page_content = text
        self.metadata = {"source": source}


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return _FakeResponse(self.reply)


def test_generate_returns_refusal_when_no_documents():
    generator = AnswerGenerator(llm=_FakeLLM("unused"))
    answer = generator.generate("What is X?", documents=[])
    assert answer.refused
    assert answer.confidence == 0.0


def test_generate_parses_citations_from_a_normal_answer():
    llm = _FakeLLM("The answer is 42 [1].")
    generator = AnswerGenerator(llm=llm)
    docs = [_FakeDoc("the source text", "a.md")]

    answer = generator.generate("What is the answer?", documents=docs)

    assert not answer.refused
    assert answer.sources == ["a.md"]
    assert answer.confidence == 1.0


def test_generate_detects_refusal_sentinel():
    llm = _FakeLLM(REFUSAL_SENTINEL)
    generator = AnswerGenerator(llm=llm)
    docs = [_FakeDoc("unrelated text", "a.md")]

    answer = generator.generate("What is the answer?", documents=docs)

    assert answer.refused


def test_generate_passes_system_prompt_and_context_to_llm():
    llm = _FakeLLM("cited [1].")
    generator = AnswerGenerator(llm=llm, system_prompt="Custom prompt {sentinel}.")
    docs = [_FakeDoc("source text", "a.md")]

    generator.generate("question", documents=docs)

    system_message = llm.last_messages[0]
    assert system_message[0] == "system"
    assert REFUSAL_SENTINEL in system_message[1]
    assert "{sentinel}" not in system_message[1]

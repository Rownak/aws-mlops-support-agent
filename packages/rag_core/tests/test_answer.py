"""Tests for answer generation: pure formatting, plus the prompt-override seam.

The real LLM call is checked manually via `rag-ask` (small evals over
mocked-API tests); here the model is a fake that records what it was sent.
"""

from rag_core.config import ChunkingConfig, RagConfig, RetrievalConfig
from rag_core.generation.answer import format_context, format_sources, generate_answer
from rag_core.generation.prompts import DEFAULT_PROMPTS, AnswerPrompts
from rag_core.retrieval.retriever import RetrievedChunk

CFG = RagConfig(
    openai_api_key="k",
    openai_chat_model="gpt-4o-mini",
    openai_embedding_model="text-embedding-3-small",
    pinecone_api_key="k",
    pinecone_index_name="i",
    pinecone_index_metric="cosine",
    aws_region="us-east-1",
    project="p",
    chunking=ChunkingConfig(),
    retrieval=RetrievalConfig(),
    sources=(),
)


def _chunk(n):
    return RetrievedChunk(
        text=f"excerpt text {n}",
        score=0.5,
        source_id="codebuild",
        source_file=f"file{n}.md",
        heading=f"Heading {n}",
        url=f"https://docs.aws.amazon.com/{n}.html",
    )


class FakeLLM:
    """Records the messages it was given and returns a canned reply."""

    def __init__(self, reply="the model's answer"):
        self.reply = reply
        self.messages = None

    def invoke(self, messages):
        self.messages = messages

        class Response:
            content = self.reply

        return Response()


# --- pure formatting ---


def test_context_numbers_and_labels_each_excerpt():
    context = format_context([_chunk(1), _chunk(2)])
    assert "[1] source: codebuild | section: Heading 1" in context
    assert "[2] source: codebuild | section: Heading 2" in context
    assert "url: https://docs.aws.amazon.com/1.html" in context
    assert "excerpt text 2" in context
    # [1] must come before [2] — citation numbers reflect retrieval rank.
    assert context.index("[1]") < context.index("[2]")


def test_sources_numbering_matches_context_numbering():
    sources = format_sources([_chunk(1), _chunk(2)])
    assert sources.startswith("Sources:")
    assert "[1] Heading 1 — https://docs.aws.amazon.com/1.html" in sources
    assert "[2] Heading 2 — https://docs.aws.amazon.com/2.html" in sources


def test_empty_chunks_produce_empty_context():
    assert format_context([]) == ""


# --- the generation call ---


def test_answer_appends_our_sources_list_not_the_models():
    """URLs come from chunk metadata, so they are real even if the model mis-cites."""
    llm = FakeLLM("See [1].")
    answer = generate_answer("q", [_chunk(1)], CFG, llm=llm)
    assert answer.startswith("See [1].")
    assert "[1] Heading 1 — https://docs.aws.amazon.com/1.html" in answer


def test_default_prompts_are_used_when_none_are_given():
    llm = FakeLLM()
    generate_answer("how do I cache?", [_chunk(1)], CFG, llm=llm)
    (system_role, system_text), (user_role, user_text) = llm.messages
    assert system_role == "system"
    assert system_text == DEFAULT_PROMPTS.system
    assert user_role == "user"
    # The template's two fields were filled in.
    assert "how do I cache?" in user_text
    assert "excerpt text 1" in user_text


def test_project_can_override_the_prompts():
    prompts = AnswerPrompts(
        system="You are an AWS CI/CD support assistant.",
        user_template="Q: {question}\nDocs: {context}",
    )
    llm = FakeLLM()
    generate_answer("how do I cache?", [_chunk(1)], CFG, prompts=prompts, llm=llm)
    (_, system_text), (_, user_text) = llm.messages
    assert system_text == "You are an AWS CI/CD support assistant."
    assert user_text.startswith("Q: how do I cache?")
    assert "Docs: [1] source: codebuild" in user_text


def test_overriding_prompts_does_not_change_the_sources_list():
    prompts = AnswerPrompts(system="custom", user_template="{question} {context}")
    answer = generate_answer("q", [_chunk(1)], CFG, prompts=prompts, llm=FakeLLM())
    assert "Sources:" in answer

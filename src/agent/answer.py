"""Task 2.2 — Generate a cited answer from the question + retrieved chunks.

Split into a pure part (`format_context`, `format_sources` — unit-testable,
no network) and the one LLM call (`generate_answer`). The "Sources" list is
built from chunk metadata by our code, NOT by the model, so the URLs are real
even if the model mis-cites.
"""

from langchain_openai import ChatOpenAI

from src.agent.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_TEMPLATE
from src.agent.retriever import RetrievedChunk
from src.config import Config


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Number the excerpts [1]..[n] and label each with its provenance.

    The numbers are what the model cites, and they match the order of the
    Sources list, so [2] in the answer always points at sources[1].
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] service: {chunk.service} | section: {chunk.heading}\n"
            f"    url: {chunk.url}\n"
            f"{chunk.text}"
        )
    return "\n\n".join(blocks)


def format_sources(chunks: list[RetrievedChunk]) -> str:
    """The [n] heading — url list appended verbatim after the model's answer."""
    lines = [f"[{i}] {chunk.heading} — {chunk.url}" for i, chunk in enumerate(chunks, start=1)]
    return "Sources:\n" + "\n".join(lines)


def generate_answer(question: str, chunks: list[RetrievedChunk], cfg: Config) -> str:
    """One chat completion: system prompt + question + numbered excerpts."""
    # temperature=0 -> as deterministic as the API allows; for doc-grounded
    # support answers we want repeatability, not creativity.
    llm = ChatOpenAI(model=cfg.openai_chat_model, temperature=0, api_key=cfg.openai_api_key)
    user_message = ANSWER_USER_TEMPLATE.format(question=question, context=format_context(chunks))
    # LangChain accepts (role, content) tuples; .content is the reply text.
    response = llm.invoke([("system", ANSWER_SYSTEM_PROMPT), ("user", user_message)])
    return f"{response.content}\n\n{format_sources(chunks)}"

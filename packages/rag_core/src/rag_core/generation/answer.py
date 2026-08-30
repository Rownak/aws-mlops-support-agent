"""Generate a cited answer from the question + retrieved chunks.

Split into a pure part (`format_context`, `format_sources` — unit-testable, no
network) and the one LLM call (`generate_answer`). The "Sources" list is built
from chunk metadata by our code, NOT by the model, so the URLs are real even
if the model mis-cites.
"""

from langchain_openai import ChatOpenAI
from langsmith import traceable

from rag_core.config import RagConfig
from rag_core.generation.prompts import DEFAULT_PROMPTS, AnswerPrompts
from rag_core.retrieval.retriever import RetrievedChunk


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Number the excerpts [1]..[n] and label each with its provenance.

    The numbers are what the model cites, and they match the order of the
    Sources list, so [2] in the answer always points at sources[1].
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] source: {chunk.source_id} | section: {chunk.heading}\n"
            f"    url: {chunk.url}\n"
            f"{chunk.text}"
        )
    return "\n\n".join(blocks)


def format_sources(chunks: list[RetrievedChunk]) -> str:
    """The [n] heading — url list appended verbatim after the model's answer."""
    lines = [f"[{i}] {chunk.heading} — {chunk.url}" for i, chunk in enumerate(chunks, start=1)]
    return "Sources:\n" + "\n".join(lines)


def _drop_cfg(inputs: dict) -> dict:
    """Config carries API keys — it must never be serialized into a trace."""
    return {k: v for k, v in inputs.items() if k != "cfg"}


# Traced so the final cited answer is the span's output, with the ChatOpenAI
# call nested inside it. No-op unless LANGSMITH_TRACING is on.
@traceable(process_inputs=_drop_cfg)
def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    cfg: RagConfig,
    prompts: AnswerPrompts = DEFAULT_PROMPTS,
    llm=None,
) -> str:
    """One chat completion: system prompt + question + numbered excerpts.

    `prompts` lets a project swap in its own voice; `llm` is injectable so
    tests can exercise this path without an API key.
    """
    if llm is None:
        # temperature=0 -> as deterministic as the API allows; for
        # doc-grounded support answers we want repeatability, not creativity.
        llm = ChatOpenAI(model=cfg.openai_chat_model, temperature=0, api_key=cfg.openai_api_key)
    user_message = prompts.user_template.format(question=question, context=format_context(chunks))
    # LangChain accepts (role, content) tuples; .content is the reply text.
    response = llm.invoke([("system", prompts.system), ("user", user_message)])
    return f"{response.content}\n\n{format_sources(chunks)}"

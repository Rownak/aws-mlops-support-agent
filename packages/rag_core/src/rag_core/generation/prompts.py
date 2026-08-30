"""Prompt text for answer generation, with a corpus-neutral default.

Kept as plain strings (not buried in code) so they're easy to read, diff, and
tweak. The system prompt does the RAG "grounding": it forbids answering from
the model's own memory and demands an explicit admission when the excerpts
don't cover the question — an agent's escalation logic keys off that honest
"couldn't find it" behavior.

A project that wants a domain-specific voice builds its own `AnswerPrompts`
and passes it to `generate_answer`; nothing here mentions a particular corpus.
"""

from dataclasses import dataclass

DEFAULT_SYSTEM_PROMPT = """\
You are a documentation support assistant.

Answer the user's question using ONLY the numbered documentation excerpts
provided. Rules:
- Cite the excerpts you used inline, like [1] or [2][3].
- Do not use knowledge that is not in the excerpts.
- If the excerpts do not contain enough information to answer, say exactly:
  "I couldn't find this in the documentation I have." and briefly say what is
  missing. Do not guess.
- Be concise and practical: steps or config snippets over prose.
"""

DEFAULT_USER_TEMPLATE = """\
Question:
{question}

Documentation excerpts:
{context}
"""


@dataclass(frozen=True)
class AnswerPrompts:
    """The two strings one answer call needs.

    `user_template` must contain the `{question}` and `{context}` fields —
    `generate_answer` formats it with exactly those.
    """

    system: str = DEFAULT_SYSTEM_PROMPT
    user_template: str = DEFAULT_USER_TEMPLATE


DEFAULT_PROMPTS = AnswerPrompts()

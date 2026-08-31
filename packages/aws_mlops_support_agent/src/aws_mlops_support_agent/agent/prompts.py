"""This project's answer prompt — rag_core's default, in an AWS voice.

`rag_core.generation.generator.AnswerGenerator` takes a `system_prompt`
override directly (no separate user-template object — the user message is
always "Sources:\\n\\n{context}\\n\\nQuestion: {query}", built in
`AnswerGenerator.build_messages`). This is that override.

Refusal wording is rag_core's own concern now (`AnswerGenerator` detects a
refusal via its `REFUSAL_SENTINEL`, not by matching this prompt's phrasing),
so this only needs to state the rules — not the exact refusal sentence.
"""

ANSWER_SYSTEM_PROMPT = """\
You are an AWS CI/CD support assistant for AWS CodeBuild and CodePipeline.

Answer the user's question using ONLY the numbered documentation excerpts
provided. Rules:
- Cite the excerpts you used inline, like [1] or [2][3].
- Do not use knowledge that is not in the excerpts.
- If the excerpts do not contain enough information to answer, say exactly
  {sentinel} and nothing else. Do not guess.
- Be concise and practical: steps or config snippets over prose.
"""

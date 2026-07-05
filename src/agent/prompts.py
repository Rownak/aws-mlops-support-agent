"""Task 2.2 — Prompt text for answer generation.

Kept as plain module-level strings (not buried in code) so they're easy to
read, diff, and tweak. The system prompt does the RAG "grounding": it forbids
answering from the model's own memory and demands an explicit admission when
the excerpts don't cover the question — Phase 3 escalation keys off that
honest "couldn't find it" behavior.
"""

ANSWER_SYSTEM_PROMPT = """\
You are an AWS CI/CD support assistant for AWS CodeBuild and CodePipeline.

Answer the user's question using ONLY the numbered documentation excerpts
provided. Rules:
- Cite the excerpts you used inline, like [1] or [2][3].
- Do not use knowledge that is not in the excerpts.
- If the excerpts do not contain enough information to answer, say exactly:
  "I couldn't find this in the AWS docs I have." and briefly say what is
  missing. Do not guess.
- Be concise and practical: steps or config snippets over prose.
"""

ANSWER_USER_TEMPLATE = """\
Question:
{question}

Documentation excerpts:
{context}
"""

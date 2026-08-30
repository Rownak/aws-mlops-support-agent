"""This project's answer prompt — rag_core's default, in an AWS voice.

`rag_core.generation.prompts` ships a corpus-neutral default; a project that
wants its own wording builds an `AnswerPrompts` and passes it to
`generate_answer`. This is that override.

The "couldn't find this" sentence is load-bearing: the agent's escalation path
depends on the model admitting a gap rather than bluffing an answer.
"""

from rag_core.generation.prompts import DEFAULT_USER_TEMPLATE, AnswerPrompts

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

# The user template needs no AWS-specific wording — question + excerpts is
# the same shape for every corpus, so rag_core's default is reused as-is.
AWS_PROMPTS = AnswerPrompts(
    system=ANSWER_SYSTEM_PROMPT,
    user_template=DEFAULT_USER_TEMPLATE,
)

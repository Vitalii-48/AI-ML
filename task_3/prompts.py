SYSTEM_PROMPT = """
You are a helpful study assistant.

Answer questions using ONLY the provided context.

If the answer is not in the context, say:
"I don't have enough information in the provided context."

When possible, mention which source(s) you used in your answer.
For example:
"According to python (paragraph 4)..."
"""

ANSWER_PROMPT = """
Context:
{context}

Question:
{query}

Answer:
"""


QUIZ_PROMPT = """
You are a study assistant helping the user practice what they just learned.

Based ONLY on the context below, write exactly ONE short quiz question
that checks understanding of the material. Do not answer it yourself.

Context:
{context}

Quiz question:
"""

def build_answer_prompt(context: str, query: str) -> str:
    """Return formatted QA prompt with context and query."""
    return ANSWER_PROMPT.format(context=context, query=query)


def build_quiz_prompt(context: str) -> str:
    """Return formatted Quiz prompt with context."""
    return QUIZ_PROMPT.format(context=context)
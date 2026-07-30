# task_3\llm.py
import os
from typing import List, Tuple

from dotenv import load_dotenv
from groq import Groq
from groq.types.chat import ChatCompletionUserMessageParam

from constants import DEFAULT_LLM_MODEL
from vector_store import VectorStore

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Create a .env file with "
        "GROQ_API_KEY=<your key> next to this script."
    )
client = Groq(api_key=api_key)


def build_context(store: VectorStore, search_results: List[Tuple[int, float]]) -> str:
    """Generates textual context from top search results — with source numbering."""
    context_parts: list[str] = []
    for rank, (idx, _) in enumerate(search_results, start=1):
        doc = store.get_by_id(idx)
        context_parts.append(
            f"Source {rank}\n"
            f"File: {doc['source']}\n"
            f"Paragraph: {doc['chunk']}\n"
            f"{doc['text']}"
        )
    return "\n\n".join(context_parts)


def generate_answer(
    store: VectorStore,
    query: str,
    search_results: List[Tuple[int, float]],
    model: str = DEFAULT_LLM_MODEL,
) -> str | None:
    """Generate an answer using retrieved documents."""
    context = build_context(store, search_results)

    prompt = f"""
You are a helpful study assistant.

Answer the question using ONLY the provided context.

If the answer is not in the context, say:
"I don't have enough information in the provided context."

When possible, mention which source(s) you used in your answer.
For example:
"According to python (paragraph 4)..."

Context:
{context}

Question:
{query}

Answer:
"""

    messages: list[ChatCompletionUserMessageParam] = [
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
    except Exception as e:
        return f"Error while contacting LLM: {e}"

    return response.choices[0].message.content


def generate_quiz_question(
    store: VectorStore,
    search_results: List[Tuple[int, float]],
    model: str = DEFAULT_LLM_MODEL,
) -> str | None:
    """Generate one short quiz question based on the top retrieved match (Study Mode)."""
    top_result = search_results[:1]
    context = build_context(store, top_result)

    prompt = f"""
You are a study assistant helping the user practice what they just learned.

Based ONLY on the context below, write exactly ONE short quiz question
that checks understanding of the material. Do not answer it yourself.

Context:
{context}

Quiz question:
"""

    messages: list[ChatCompletionUserMessageParam] = [
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
    except Exception as e:
        return f"Error while generating quiz question: {e}"

    return response.choices[0].message.content
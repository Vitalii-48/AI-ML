# task_3\llm.py
import os

from dotenv import load_dotenv
from groq import Groq
from groq.types.chat import ChatCompletionUserMessageParam

from constants import DEFAULT_LLM_MODEL
from prompts import build_answer_prompt, build_quiz_prompt
from vector_store import VectorStore

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Create a .env file with "
        "GROQ_API_KEY=<your key> next to this script."
    )
client = Groq(api_key=api_key)


def build_context(store: VectorStore, search_results: list[tuple[int, float]]) -> str:
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
    search_results: list[tuple[int, float]],
    model: str = DEFAULT_LLM_MODEL,
) -> str | None:
    """Generate an answer using retrieved documents."""
    context = build_context(store, search_results)
    prompt = build_answer_prompt(context=context, query=query)

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
    search_results: list[tuple[int, float]],
    model: str = DEFAULT_LLM_MODEL,
) -> str | None:
    """Generate one short quiz question based on the top retrieved match (Study Mode)."""
    top_result = search_results[:1]
    context = build_context(store, top_result)

    prompt = build_quiz_prompt(context=context)

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
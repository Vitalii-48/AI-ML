from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from groq import Groq
from groq.types.chat import ChatCompletionUserMessageParam

from constants import DEFAULT_LLM_MODEL
from enums import Role
from prompts import build_answer_prompt, build_quiz_prompt
from vector_store import VectorStore


def build_context(store: VectorStore, search_results: list[tuple[int, float]]) -> str:
    """Build a numbered-source text context from the top search results."""
    context_parts: list[str] = []
    for rank, (idx, _) in enumerate(search_results, start=1):
        doc = store.get_by_id(idx)
        context_parts.append(
            f"Source {rank}\n"
            f"File: {doc.source}\n"
            f"Paragraph: {doc.chunk}\n"
            f"{doc.text}"
        )
    return "\n\n".join(context_parts)


def _call_llm(client: Groq, prompt: str, model: str) -> str:
    """Send a single-message prompt to Groq and return the reply text.

    Raises RuntimeError with a clear, specific message for each known
    failure mode, instead of catching a bare Exception.
    """
    messages: list[ChatCompletionUserMessageParam] = [
        {"role": Role.USER.value, "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
    except AuthenticationError as exc:
        raise RuntimeError("Authentication failed: check your GROQ_API_KEY.") from exc
    except RateLimitError as exc:
        raise RuntimeError("Rate limit hit (429): please wait a moment and try again.") from exc
    except APITimeoutError as exc:
        raise RuntimeError("The request timed out. Check your connection and try again.") from exc
    except APIConnectionError as exc:
        raise RuntimeError(f"Connection error while calling Groq API: {exc}") from exc
    except APIStatusError as exc:
        raise RuntimeError(f"Groq API returned an error (status {exc.status_code}): {exc.message}") from exc

    return response.choices[0].message.content or ""


def generate_answer(
    client: Groq,
    store: VectorStore,
    query: str,
    search_results: list[tuple[int, float]],
    model: str = DEFAULT_LLM_MODEL,
) -> str:
    """Generate an answer to `query` using only the retrieved documents."""
    context = build_context(store, search_results)
    prompt = build_answer_prompt(context=context, query=query)
    return _call_llm(client, prompt, model)


def generate_quiz_question(
    client: Groq,
    store: VectorStore,
    search_results: list[tuple[int, float]],
    model: str = DEFAULT_LLM_MODEL,
) -> str:
    """Generate one short quiz question based on the single top match (Study Mode)."""
    top_result = search_results[:1]
    context = build_context(store, top_result)

    prompt = build_quiz_prompt(context=context)
    return _call_llm(client, prompt, model)
import os

from dotenv import load_dotenv
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
from prompts import build_answer_prompt, build_quiz_prompt
from vector_store import VectorStore


class LLMService:
    def __init__(self, model: str = DEFAULT_LLM_MODEL) -> None:
        self.model = model
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Create a .env file with "
                "GROQ_API_KEY=<your key> next to this script."
            )

        self.client = Groq(api_key=api_key)

    @staticmethod
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

    def _call_llm(self, prompt: str) -> str:
        """Send a single-message prompt to Groq and return the reply text.

        Raises RuntimeError with a clear, specific message for each known
        failure mode, instead of catching a bare Exception.
        """
        messages: list[ChatCompletionUserMessageParam] = [
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
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
            self,
            store: VectorStore,
            query: str,
            search_results: list[tuple[int, float]],
    ) -> str:
        """Generate an answer to `query` using only the retrieved documents."""
        context = self.build_context(store, search_results)
        prompt = build_answer_prompt(context=context, query=query)
        return self._call_llm(prompt)

    def generate_quiz_question(
            self,
            store: VectorStore,
            search_results: list[tuple[int, float]],
    ) -> str:
        """Generate one short quiz question based on the single top match (Study Mode)."""
        top_result = search_results[:1]
        context = self.build_context(store, top_result)
        prompt = build_quiz_prompt(context=context)
        return self._call_llm(prompt)


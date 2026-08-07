from pathlib import Path

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from groq import Groq

from constants import DEFAULT_LLM_MODEL, DEFAULT_WHISPER_MODEL
from prompts import build_summary_prompt
from settings import settings


class AudioLLMService:
    """Handles Whisper transcription and LLM-based transcript summarization via Groq."""

    def __init__(
        self,
        llm_model: str = DEFAULT_LLM_MODEL,
        whisper_model: str = DEFAULT_WHISPER_MODEL,
    ) -> None:
        self.llm_model = llm_model
        self.whisper_model = whisper_model
        self.client = Groq(api_key=settings.groq_api_key)

    def _handle_api_exception(self, exc: Exception) -> None:
        if isinstance(exc, AuthenticationError):
            raise RuntimeError("Authentication failed: check your GROQ_API_KEY.") from exc
        elif isinstance(exc, RateLimitError):
            raise RuntimeError("Rate limit hit (429): please wait a moment and try again.") from exc
        elif isinstance(exc, APITimeoutError):
            raise RuntimeError("The request timed out. Check your connection and try again.") from exc
        elif isinstance(exc, APIConnectionError):
            raise RuntimeError(f"Connection error while calling Groq API: {exc}") from exc
        elif isinstance(exc, APIStatusError):
            raise RuntimeError(f"Groq API returned an error (status {exc.status_code}): {exc.message}") from exc
        else:
            raise exc

    def transcribe(self, file_path: Path) -> str:
        """Transcribe a local audio file to text using Groq Whisper."""
        try:
            with open(file_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    file=audio_file,
                    model=self.whisper_model,
                )
        except Exception as exc:
            self._handle_api_exception(exc)
        return transcription.text

    def summarize(self, transcript: str) -> str:
        """Summarize a transcript into a short summary and bullet-point key points."""
        messages = [
            {
                "role": "user",
                "content": build_summary_prompt(transcript),
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.3,
            )
        except Exception as exc:
            self._handle_api_exception(exc)

        content = response.choices[0].message.content
        return content if content is not None else ""
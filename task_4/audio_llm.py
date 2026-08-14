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
from prompts import build_summary_prompt, build_keywords_prompt, build_title_prompt, build_qna_prompt
from settings import settings
from enums import Mode


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
        except (
            AuthenticationError,
            RateLimitError,
            APITimeoutError,
            APIConnectionError,
            APIStatusError,
        ) as exc:
            self._handle_api_exception(exc)

        return transcription.text

    def process_transcript(self, transcript: str, mode: Mode) -> str:
        """Process a transcript according to the selected mode."""
        if mode == Mode.SUMMARY:
            prompt = build_summary_prompt(transcript)
        elif mode == Mode.EXTRACT_KEYWORDS:
            prompt = build_keywords_prompt(transcript)
        elif mode == Mode.GENERATE_TITLE:
            prompt = build_title_prompt(transcript)
        elif mode == Mode.QNA:
            prompt = build_qna_prompt(transcript)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.3,
                response_format={"type": "json_object"},
            )
        except (
                AuthenticationError,
                RateLimitError,
                APITimeoutError,
                APIConnectionError,
                APIStatusError,
        ) as exc:
            self._handle_api_exception(exc)

        content = response.choices[0].message.content
        return content if content is not None else ""
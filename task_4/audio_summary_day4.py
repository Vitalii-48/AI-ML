from pathlib import Path

from pydantic import ValidationError

from audio_llm import AudioLLMService
from constants import (
    DEFAULT_LLM_MODEL,
    DEFAULT_WHISPER_MODEL,
    AUDIO_DIR
)


def main():
    try:
        service = AudioLLMService(
            llm_model=DEFAULT_LLM_MODEL,
            whisper_model=DEFAULT_WHISPER_MODEL,
        )
    except ValidationError:
        print("[error] GROQ_API_KEY is not set. Check your .env file.")
        return

    print(f"Audio Assistant {DEFAULT_LLM_MODEL}")
    print()
    print("Hi! I'm your audio assistant.")
    print("I can transcribe and analyze audio files.")
    print("Enter the name of the audio files from the 'audio' folder")

    while True:
        query = input("> You ").strip()

        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if not query:
            print("Please enter an audio filename from the 'audio' folder.\n")
            continue

        file_path = AUDIO_DIR / query
        if not file_path.is_file():
            print("Assistant: Please provide a valid audio file.")
            continue

        try:
            transcript = service.transcribe(file_path)
            print(transcript)

            summary = service.summarize(transcript)
            print(summary)

        except RuntimeError as exc:
            print(f"\n[error] {exc}\n")
            continue


if __name__ == "__main__":
    main()

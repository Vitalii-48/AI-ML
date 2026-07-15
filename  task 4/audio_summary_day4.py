# task 4\audio_summary_day4.py

import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent

def transcribe_audio(client: Groq, file_path: str) -> str:
    """Sends an audio file to Groq Whisper and returns plain text."""
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(file_path, audio_file.read()),
            model="whisper-large-v3-turbo",
            response_format="text"
        )
    return str(transcription).strip()


def summarize_transcription(client: Groq, transcript: str) -> str | None:
    """Takes transcript text and generates a structured Summary using LLM."""
    system_prompt = (
        "You are a helpful assistant that summarizes audio transcripts. "
        "Provide a concise summary and list the key points in bullet points. "
        "Respond in the language of the transcript (or English if requested)."
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[                                      # type: ignore
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please summarize the following transcript:\n\n{transcript}"}
            ],
            temperature=0.3
        )
    except Exception as e:
        return  f"Error while contacting LLM: {e}"

    return response.choices[0].message.content


def save_to_markdown_log(file_path: Path, transcript: str, summary: str | None) -> Path:
    """Saves audio file to Markdown log."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = f"{timestamp}_{file_path.stem}.md"
    log_path = logs_dir / log_filename

    markdown_content = f"""# Audio Processing Report: {file_path.name}

- **Processing Date and Time:** {datetime.now().strftime("%d.%m.%Y %H:%M:%S")}
- **Original File:** `{file_path.name}`
- **File Size:** {file_path.stat().st_size / 1024 / 1024:.2f} MB

---

## Full Transcript (Whisper)

{transcript}

---

## Summary (LLM)

{summary}
"""
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return log_path



def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Create a .env file with "
            "GROQ_API_KEY=<your key> next to this script."
        )
    client = Groq(api_key=api_key)

    test_file = SCRIPT_DIR / "test_audio.mp3"


    print("================================================================")
    print("CLI Audio Assistant: Transcription and Summarization️")
    print("================================================================")
    print("Enter the full or relative path to the file (for example: test_audio.mp3).")
    print("To exit the program, type 'exit'.\n")

    while True:
        user_input = input("\n> Enter the path to the audio file (or 'exit'): ").strip()

        if user_input.lower() in ("exit", "quit"):
            print("\nGoodbye!")
            break

        if not user_input:
            print("Please enter the file path.\n")
            continue

        file_path = Path(user_input)

        if not file_path.exists():
            print("=== audiofile does not exist ===")
            continue

        try:
            transcript = transcribe_audio(client, str(file_path))
            print("\n=== [1] WHISPER TRANSCRIPT ===")
            print(transcript)
            print("==============================\n")

            if not transcript:
                print("The audio file is empty or the model did not recognize any words.")
                continue

            print(f"2. Sending text for autorization to Llama...")
            summary = summarize_transcription(client, transcript)
            print("=== [2] LLM SUMMARY ===")
            print(summary)
            print("=======================")

            log_file = save_to_markdown_log(file_path, transcript, summary)
            print(f"\n[Logs] Results successfully saved to report: {log_file}\n")
        except Exception as e:
            print(f"An error occurred:: {e}")

if __name__ == "__main__":
    main()

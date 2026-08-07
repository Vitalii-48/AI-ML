# Task 4 — Audio Transcription & Summarization Assistant

A command-line audio assistant powered by the **Groq API**. The application
transcribes local audio files using **Whisper** and generates concise summaries
using a large language model. This task extends the previous AI assistant by
adding speech-to-text capabilities.

## File structure

```text
task_4/
├── audio_summary_day4.py    # CLI entry point
├── audio_llm.py             # AudioLLMService (Whisper + LLM)
├── settings.py              # Pydantic Settings
├── constants.py             # Centralized configuration
├── prompts.py               # Prompt templates & builder functions
├── audio/                   # Input audio files
├── requirements.txt
└── README.md
```

## Setup

1. Install project dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root and add your Groq API key:

```text
GROQ_API_KEY=your_groq_api_key_here
```

## Running the application

```bash
python audio_summary_day4.py
```

## Conversation flow

1. Place an audio file (`.mp3`, `.wav`, or `.m4a`) into the `audio/` folder.
2. Enter the audio filename in the CLI.
3. The application transcribes the audio using **Groq Whisper**.
4. The transcript is displayed in the terminal.
5. The transcript is summarized using a Groq language model.
6. The assistant continues running until the user enters `exit`, `quit`, or `q`.

## Core Features

- ✅ Interactive CLI
- ✅ Audio transcription using Whisper (`whisper-large-v3`)
- ✅ Transcript summarization with Groq LLM
- ✅ Supports `.mp3`, `.wav`, and `.m4a` files
- ✅ Pydantic Settings for configuration management
- ✅ Centralized prompts and constants
- ✅ `pathlib.Path` for file handling
- ✅ Graceful API error handling

## Error handling

The application handles common runtime errors, including:

- missing `GROQ_API_KEY`;
- invalid or missing audio file;
- authentication errors;
- rate limit errors;
- network connection errors;
- API timeout errors;
- Groq API status errors.

## Example session

```text
Audio Assistant llama-3.3-70b-versatile

Hi! I'm your audio assistant.
I can transcribe and analyze audio files.
Place your audio files in the "audio" folder and enter the filename.

> You: lecture.mp3

Transcript:
Machine learning models can generalize better when trained on diverse datasets...

Summary:

The recording explains how dataset diversity improves model generalization.
It discusses the importance of representative training data and reducing bias.

Key points:

- Diverse datasets improve generalization.
- Better coverage reduces model bias.
- Representative data leads to more robust predictions.

> You: exit

Goodbye!
```
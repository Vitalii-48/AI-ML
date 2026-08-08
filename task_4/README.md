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
├── enums.py                 # Mode enum (summary/extract_keywords/generate_title/qna)
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

```bash
python audio_summary_day4.py --mode qna
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
- ✅ Multiple processing modes (summary, extract_keywords, generate_title, qna)

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
Enter the name of the audio files from the 'audio' folder, for example test_audio1.mp3
> You: test_audio1.mp3
 Hi. I'm doing an internship. I need to create a test audio file for my fourth task. Could you generate an English voice recording of this text?
result of work mode: qna
Q1: What is the speaker currently doing?
A1: The speaker is doing an internship.

Q2: What is the speaker's current task?
A2: The speaker's current task is their fourth task.

Q3: What does the speaker need to create for their task?
A3: The speaker needs to create a test audio file.

Q4: What language should the voice recording be in?
A4: The voice recording should be in English.
> You: exit
Goodbye!
```
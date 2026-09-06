# Task 5 Capstone — CLI AI Assistant with Memory, Tools, and Voice Input

A modular command-line assistant built with the Groq API. It supports persistent
conversation memory, a semantic (embeddings-based) knowledge base, text and voice
input, function calling, and session save/load.

## Features

- **Conversational memory** — full chat history is kept and sent with every request, with streaming responses printed
  token-by-token.
- **Semantic knowledge base** — facts are stored as embeddings (`sentence-transformers`, `all-MiniLM-L6-v2`) and
  retrieved by meaning, not by keyword overlap.
- **Voice input** — audio files (`.mp3`, `.wav`, `.m4a`) are transcribed with Groq's Whisper and added to the knowledge
  base with source metadata.
- **Function calling** — the assistant can call two tools on its own (or on explicit command):
    - `semantic_search` — looks up the most relevant fact in the knowledge base.
    - `summarize_session` — returns a clean bullet-point summary of the conversation.
- **Configurable model** — switch the underlying Groq model via a CLI flag.
- **Editable system prompt** — change the assistant's persona at any time, from raw text or a file.
- **Session persistence** — save and reload the full conversation + knowledge base.

## Project Structure

```
.
├── requirements.txt         # Root dependencies file
└── Task5_Capstone/
    ├── main.py              # CLI loop, commands, orchestration
    ├── tools.py             # Tool schemas + search_kb() / summarize_session
    ├── vector_store.py      # VectorStore class
    ├── prompts.py           # SYSTEM_PROMPT and SUMMARY_PROMPT text
    ├── constants.py         # Centralized paths and model names
    ├── audio/               # sample audio files
    ├── sessions/            # saved session .json files
    └── README.md
```

## Setup

1. Install dependencies:
   ```
   pip install groq python-dotenv sentence-transformers numpy
   ```
2. Create a `.env` file in this folder with your Groq API key:
   ```
   GROQ_API_KEY=your_key_here
   ```
3. Run the assistant:
   ```
   python main.py
   ```

   Optionally choose a different model:
   ```
   python main.py -model llama-3.1-8b-instant
   ```

## Commands

| Command              | Description                                                                                                       |
|----------------------|-------------------------------------------------------------------------------------------------------------------|
| `/update_kb_text`    | Add a fact to the knowledge base by typing it.                                                                    |
| `/update_kb_voice`   | Add a fact by transcribing an audio file (`.mp3`/`.wav`/`.m4a`).                                                  |
| `/search`            | Force a semantic search over the knowledge base for a given query.                                                |
| `/summarize_session` | Force a bullet-point summary of the conversation so far.                                                          |
| `/change_prompt`     | Change the assistant's system prompt — enter new text directly, or a path to a `.txt` file containing the prompt. |
| `/save_session`      | Save the current conversation and knowledge base to a timestamped `.json` file in `sessions/`.                    |
| `/load_session`      | Load a previously saved session file, restoring both conversation and knowledge base.                             |
| `/exit`              | Exit the assistant.                                                                                               |

Any other input is treated as a normal chat message. The assistant may decide on
its own to call `semantic_search` or `summarize_session` if it judges that
appropriate, based on its system prompt.

## Example Run

```
> You: /update_kb_text
Enter your fact: I have a cat named Luna.
[Saved] Fact saved. Total facts in KB: 1

> You: What is my cat's name?
Assistant: Your cat's name is Luna.

> You: /update_kb_voice
Enter audio file path (e.g. test_audio1.mp3): test_audio1.mp3
[Transcribed] "..."
[Saved] Fact saved from voice (source: test_audio1.mp3). Total facts in KB: 2

> You: /summarize_session
Assistant: * The user shared that their cat is named Luna.
* ...

> You: /save_session
Assistant: Session saved to session_2026-07-17_14-36-06.json

> You: /exit
Exiting...
```

## Design Notes

- **`tools.py` holds both halves of each tool**: the JSON schema the model sees
  (`SEMANTIC_SEARCH_TOOL`, `SUMMARIZE_SESSION_TOOL`, combined into `tools`) and
  the Python functions that actually run when the model calls them (`search_kb`,
  `summarize_session`). Keeping them together makes it obvious which schema
  corresponds to which implementation.
- **Knowledge base vs. conversation memory** are kept separate: the knowledge base
  (`VectorStore`) holds durable facts the user has explicitly saved, while
  `messages` holds the turn-by-turn dialogue. `/summarize_session` summarizes
  only the conversation, not the knowledge base, per the assignment scope.
- **All prompt text lives in `prompts.py`**, and all model names / file paths
  live in `constants.py` — nothing is hardcoded inline, so either can be changed
  in one place.
- **Tool calling uses `temperature=0`** to reduce (though not fully eliminate)
  occasional malformed function-call output from the underlying model — a known
  limitation of `llama-3.3-70b-versatile` on Groq.
- **Sessions are serialized without embeddings** — only text and metadata are
  saved; embeddings are recomputed on load. This keeps session files small and
  human-readable, at the cost of a short delay when loading a large knowledge base.
- **Path handling is script-relative** (`Path(__file__).resolve().parent`), so
  the assistant can be run from any working directory and still find its
  `audio/` and `sessions/` folders correctly.

## Known Limitations / Not Implemented

- Rich terminal UI (color-coded roles) — not implemented.
- Text-to-speech (`--voice` output) — not implemented.
- Persona flags (`-persona funny`) — superseded by `/change_prompt`, which covers the same use case interactively.
- `/retry last` — not implemented.
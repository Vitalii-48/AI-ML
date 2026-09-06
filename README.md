# AI-ML — Internship Projects

A series of progressively complex CLI-based AI tools built in Python using the **Groq API**,
developed during a structured AI/ML internship at Meduzzen. Each task builds on the previous one,
extending a shared `ChatSession`-style architecture with new capabilities: streaming, tool/function calling,
semantic search (RAG), audio transcription, and finally a capstone assistant combining all of them.

## Structure

| Folder | Task | What it adds |
|---|---|---|
| [`task_1`](task_1/) | Multi-turn CLI chat | Streaming responses, token counting, conversation logging |
| [`task_2`](task_2/) | Tool / function calling | Model-driven tool use (`calculate`, `explain`, `fake_lookup`, `wikipedia_search`) |
| [`task_3`](task_3/) | Semantic search (RAG) | Local embeddings (`sentence-transformers`), cosine-similarity retrieval, Study Mode quizzes |
| [`task_4`](task_4/) | Audio transcription & summarization | Groq Whisper transcription + LLM summarization, multiple processing modes |
| [`Task5_Capstone`](Task5_Capstone/) | Capstone assistant | Combines memory, tools, semantic knowledge base, voice input/output, and session persistence |

Each folder has its own `README.md` with setup instructions, sample commands, and a breakdown of implemented features — follow the links above for details.

## Common setup

All tasks share the same general setup pattern:

1. Install dependencies (per-task `requirements.txt`, or the root one for shared packages):
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. Run the relevant task's entry point from inside its folder, e.g.:
   ```bash
   python task_1/app.py
   ```

## Design principles followed throughout

- **DRY** — each task reuses and extends components from the previous one (e.g. `ChatSession` from Task 1 is extended in Task 2, and reused again in the capstone).
- **Single responsibility** — constants, prompts, tool schemas, and business logic live in separate modules rather than one large script.
- **No magic strings** — configuration values and closed sets of options live in `constants.py` / `enums.py`.
- **Atomic state updates** — conversation history is only updated after a fully successful API turn; failed calls roll back cleanly instead of leaving partial state.
- **Specific error handling** — Groq API errors (`AuthenticationError`, `RateLimitError`, `APITimeoutError`, `APIConnectionError`, `APIStatusError`)
are caught individually rather than with a bare `except Exception`.
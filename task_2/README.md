# Task 2 — Tool/Function Calling: Weather Bot & Math Assistant

A CLI educational assistant built on top of the Task 1 `ChatSession`,
extended with **tool/function calling**: the model decides on its own when
to call a local Python function before replying in natural language.

## File structure

```
task_2/
├── app.py               # CLI entry point
├── chat_session.py       # ChatSession (extended with tool calling)
├── tools.py               # Tool functions, logging, dispatch map
├── tool_schemas.py        # JSON schemas of the tools (for the LLM)
├── constants.py
├── enums.py
├── fake_db.json            # data for fake_lookup
├── logs/
└── .env.example
```

## Setup

```bash
pip install -r requirements.txt
```
Copy `.env.example` to `.env` and add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Running

```bash
python app.py
```

## Sample prompts

- `calculate 5 * (2 + 3)`
- `explain gravity`
- `look up python`
- `search Wikipedia for Ukraine`
- `explain gravity and calculate 5*3` (combined tools in one message)

## Tools

| Tool | Description |
| --- | --- |
| `calculate(expr)` | Solves math expressions |
| `explain(topic)` | Explains a topic from a local knowledge base |
| `fake_lookup(query)` *(bonus)* | Simulated encyclopedia lookup, backed by `fake_db.json` |
| `wikipedia_search(query)` *(bonus)* | Live Wikipedia search |

## How it works

1. User message + tool schemas are sent to the model.
2. If the model returns `tool_calls`, `ChatSession` runs the matching
   Python function via a dispatch map (`TOOL_FUNCTIONS`).
3. Tool results are sent back in a second request, which produces the
   final reply.
4. Conversation history (`self.messages`) is updated only after the whole
   turn succeeds — a failed API call never leaves the history half-broken.

## Logging

- Conversation log: `logs/chat_*.md` / `.json` (same as Task 1)
- Tool-call log *(bonus)*: `logs/YYYY-MM-DD.<fmt>`, one line per tool call

## Error handling

Same pattern as Task 1 (missing API key, rate limit, timeout, connection
errors), plus handling for `tool_use_failed` — an occasional model
formatting error (known limitation of `llama-3.3-70b-versatile`) that's
caught gracefully instead of crashing the session.

## Bonus features implemented

- ✅ `fake_lookup` (local JSON knowledge base)
- ✅ `wikipedia_search` (live Wikipedia API)
- ✅ Tool-call logging
- ✅ Combined queries (multiple tools per message)
- ✅ Token counting (inherited from Task 1)
- ✅ End-of-session summary
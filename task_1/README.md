# Task 1 — Multi-Turn CLI Chat with Streaming, Token Counting & Logging

A console (CLI) chat assistant that behaves like a tutor: it explains concepts,
asks follow-up questions, and checks your understanding. Built on the **Groq
API** (chosen back in Task 0), with real-time response streaming, token
counting, and persistent conversation logs.

## File structure

```
task1/
├── chat_session.py   # ChatSession class — all logic for talking to the Groq API
├── app.py             # CLI entrypoint
├── logs/               # saved conversation logs (Markdown)
│   └── 2026-07-12.md   # example saved conversation
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and paste your Groq API key (the same
   `groq-key` created back in Task 0):
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Running the app

```bash
python app.py
```
## Sample prompts to test with

- **Explain:** `Explain overfitting in simple terms.`
- **Quiz me:** `Ask me questions to test my Python knowledge.`
- **Compare:** `Compare supervised and unsupervised learning.`
- **Summarize:** `Summarize this paragraph: ...`

## How token counting works

While streaming, the Groq API attaches real usage statistics
(`prompt_tokens`, `completion_tokens`, `total_tokens`) to the **final**
streamed chunk of a response (`chunk.x_groq.usage`). This is the primary,
accurate method used for counting.

If the provider doesn't return usage data for some reason, `ChatSession`
automatically falls back to counting via `tiktoken` (the `cl100k_base`
encoding), which gives a close estimate of token count for Llama-style
models.

After every response the console prints:
```
[Tokens used: 142 | Total so far: 142]
```

## The `ChatSession` class

Located in `chat_session.py`:

- keeps the full message history (`system` / `user` / `assistant`);
- `send_message(user_input, on_chunk=...)` — sends a message, streams the
  reply piece by piece (via the `on_chunk` callback, which the CLI uses to
  print in real time), and tracks tokens per turn and per session;
- `save_log(fmt="md"|"json")` — saves the whole conversation, with
  timestamps and token stats, to `logs/{date}.md` or `logs/{date}.json`;
- `summary()` — a short end-of-session summary (turn count, tokens, model).

## Error handling

- **Missing API key** — raises a clear error when `ChatSession` is created,
  pointing you to create a `.env` file.
- **429 (rate limit)** — caught separately; the user is told to wait and
  retry, and the session doesn't crash.
- **Timeout / connection issues** — separate, readable error messages; the
  conversation can continue (you can type your next message).

## Streaming and the "typing..." indicator

Before each reply, `[Assistant is typing...]` is printed, then the
assistant's text is printed chunk by chunk, as soon as it arrives from the
API — without waiting for the full response.

## Bonus features implemented

- ✅ Token counting via real Groq usage data + `tiktoken` fallback
- ✅ Colored output via `rich` (with a graceful fallback to plain `print` if
  `rich` isn't installed)
- ✅ End-of-session summary (`summary()`) when quitting the chat
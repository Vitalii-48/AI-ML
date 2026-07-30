# task_1/chat_session.py
from __future__ import annotations

import json
import os
import tiktoken
from datetime import datetime
from pathlib import Path
from typing import Callable

from groq import Groq
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from constants import DEFAULT_SYSTEM_PROMPT, MODEL, DEFAULT_LOG_DIR
from enums import LogFormat


_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Estimate the number of tokens in `text`. """
    if not text:
        return 0
    return len(_ENCODER.encode(text))


class ChatSession:
    """Manages one multi-turn conversation with the Groq API."""

    def __init__(
        self,
        model: str = MODEL,
        system_prompt: str | None = None,
        log_dir: str = DEFAULT_LOG_DIR,
    ) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Create a .env file with "
                "GROQ_API_KEY=<your key> next to this script."
            )

        self.client = Groq(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        self.total_tokens = 0
        self.turns: list[dict] = []
        self.session_started = datetime.now()

    # ------------------------------------------------------------------
    # Conversation handling
    # ------------------------------------------------------------------
    def send_message(self, user_input: str, on_chunk: Callable[[str | None], None] | None = None) -> dict:
        """Send a user message, stream the assistant's reply, and track tokens."""
        buf_messages = self.messages + [{"role":"user", "content": user_input}]
        prompt_tokens_est = count_tokens(user_input)

        full_reply = ""
        usage = None

        try:
            stream = self.client.chat.completions.create(       # type: ignore
                messages=buf_messages,
                model=self.model,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    full_reply += delta
                    if on_chunk:
                        on_chunk(delta)
                # Groq attaches real usage stats to the final streamed chunk.
                chunk_usage = getattr(chunk, "x_groq", None)
                if chunk_usage and getattr(chunk_usage, "usage", None):
                    usage = chunk_usage.usage

        except AuthenticationError as exc:
            raise RuntimeError("Authentication failed: check your GROQ_API_KEY.") from exc
        except RateLimitError as exc:
            raise RuntimeError("Rate limit hit (429): please wait a moment and try again.") from exc
        except APITimeoutError as exc:
            raise RuntimeError("The request timed out. Check your connection and try again.") from exc
        except APIConnectionError as exc:
            raise RuntimeError(f"Connection error while calling Groq API: {exc}") from exc
        except APIStatusError as exc:
            raise RuntimeError(f"Groq API returned an error (status {exc.status_code}): {exc.message}") from exc

        self.messages.append({"role": "user", "content": user_input})
        self.messages.append({"role": "assistant", "content": full_reply})

        if usage is not None:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            turn_tokens = usage.total_tokens
        else:
            # Fallback estimate if the API didn't return usage stats.
            prompt_tokens = prompt_tokens_est
            completion_tokens = count_tokens(full_reply)
            turn_tokens = prompt_tokens + completion_tokens

        self.total_tokens += turn_tokens

        turn_record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "user": user_input,
            "assistant": full_reply,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "turn_tokens": turn_tokens,
            "total_tokens_so_far": self.total_tokens,
        }
        self.turns.append(turn_record)
        return turn_record

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_log(self, fmt: LogFormat = LogFormat.MD) -> Path:
        """Write the full conversation to logs/{date}.md or .json."""
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if fmt == LogFormat.JSON:
            path = self.log_dir / f"chat_{date_str}.json"
            payload = {
                "session_started": self.session_started.isoformat(timespec="seconds"),
                "model": self.model,
                "system_prompt": self.system_prompt,
                "total_tokens": self.total_tokens,

                "turns": self.turns,
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            path = self.log_dir / f"chat_{date_str}.md"
            lines = [
                f"# Chat log — {date_str}",
                "",
                f"**Model:** `{self.model}`  ",
                f"**System prompt:** {self.system_prompt}  ",
                f"**Session started:** {self.session_started.isoformat(timespec='seconds')}  ",
                "",
                "---",
                "",
            ]
            for turn in self.turns:
                lines.append(f"###  You ({turn['timestamp']})")
                lines.append(turn["user"])
                lines.append("")
                lines.append("###  Assistant")
                lines.append(turn["assistant"])
                lines.append("")
                lines.append(
                    f"_Tokens used: {turn['turn_tokens']} "
                    f"(prompt: {turn['prompt_tokens']}, completion: {turn['completion_tokens']}) "
                    f"| Total so far: {turn['total_tokens_so_far']}_"
                )
                lines.append("")
                lines.append("---")
                lines.append("")
            lines.append(f"**Total tokens used this session: {self.total_tokens}**")
            path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def summary(self) -> str:
        """Return a short end-of-session summary string."""
        return (
            f"Turns: {len(self.turns)} | "
            f"Total tokens used: {self.total_tokens} | "
            f"Model: {self.model}"
        )
# task_2/chat_session.py
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

import tiktoken
from groq import Groq
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from constants import DEFAULT_SYSTEM_PROMPT, LOG_DIR, MODEL
from enums import LogFormat, Role
from tool_schemas import TOOLS_SCHEMA
from tools import TOOL_FUNCTIONS, log_tool_call

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Estimate the number of tokens in `text`."""
    if not text:
        return 0
    return len(_ENCODER.encode(text))


class ChatSession:
    """Manages one multi-turn conversation with the Groq API, including
    tool/function calling."""

    def __init__(
        self,
        model: str = MODEL,
        system_prompt: str | None = None,
        log_dir: str = LOG_DIR,
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

        self.messages: list[dict] = [{"role": Role.SYSTEM.value, "content": self.system_prompt}]
        self.total_tokens = 0
        self.turns: list[dict] = []
        self.session_started = datetime.now()

    # ------------------------------------------------------------------
    # Tool calling
    # ------------------------------------------------------------------
    def _execute_tool_calls(self, tool_calls: list) -> list[dict]:
        """Run each requested tool and return the resulting 'tool' messages.

        Returns a plain list instead of mutating self.messages directly, so
        the caller can decide whether to keep these messages (only once the
        whole turn succeeds).
        """
        tool_messages = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            log_tool_call(function_name, function_args)

            tool_function = TOOL_FUNCTIONS.get(function_name)
            if tool_function is None:
                tool_result = f"Error: unknown tool '{function_name}'."
            else:
                try:
                    tool_result = tool_function(**function_args)
                except Exception as e:
                    tool_result = f"Error while running '{function_name}': {e}"

            tool_messages.append(
                {
                    "role": Role.TOOL.value,
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result,
                }
            )
        return tool_messages

    @staticmethod
    def _assistant_tool_call_message(response_message) -> dict:
        """Build a plain dict for the assistant's tool-call message."""
        return {
            "role": Role.ASSISTANT.value,
            "content": response_message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in response_message.tool_calls
            ],
        }

    # ------------------------------------------------------------------
    # Conversation handling
    # ------------------------------------------------------------------
    def send_message(self, user_input: str, on_chunk: Callable[[str | None], None] | None = None) -> dict:
        """Send a user message, letting the model call tools if it needs to,
        then stream the final reply and track tokens."""
        pending_messages = self.messages + [{"role": Role.USER.value, "content": user_input}]
        prompt_tokens_est = count_tokens(user_input)
        usage = None

        try:
            first_response = self.client.chat.completions.create(  # type: ignore
                model=self.model,
                messages=pending_messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                stream=False,
            )
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

        response_message = first_response.choices[0].message

        if response_message.tool_calls:
            pending_messages.append(self._assistant_tool_call_message(response_message))
            pending_messages.extend(self._execute_tool_calls(response_message.tool_calls))

        full_reply = ""
        try:
            stream = self.client.chat.completions.create(  # type: ignore
                model=self.model,
                messages=pending_messages,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    full_reply += delta
                    if on_chunk:
                        on_chunk(delta)
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

        # Everything succeeded — now, and only now, commit to session state.
        pending_messages.append({"role": Role.ASSISTANT.value, "content": full_reply})
        self.messages = pending_messages

        if usage is not None:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            turn_tokens = usage.total_tokens
        else:
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
        """Write the full conversation to logs/chat_{date}.md or .json."""
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
                lines.append(f"### You ({turn['timestamp']})")
                lines.append(turn["user"])
                lines.append("")
                lines.append("### Assistant")
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
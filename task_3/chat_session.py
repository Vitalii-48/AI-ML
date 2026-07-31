from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from groq import Groq

from constants import DEFAULT_LOG_DIR, DEFAULT_LOG_FORMAT, DEFAULT_LLM_MODEL
from enums import LogFormat
from llm import generate_answer, generate_quiz_question
from vector_store import VectorStore

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Estimate the number of tokens in `text`."""
    if not text:
        return 0
    return len(_ENCODER.encode(text))


class StudySession:
    """Owns the VectorStore, the Groq client, and the history of turns."""

    def __init__(
        self,
        knowledge_dir: str,
        model: str = DEFAULT_LLM_MODEL,
        log_dir: str = DEFAULT_LOG_DIR,
    ) -> None:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Create a .env file with "
                "GROQ_API_KEY=<your key> next to this script."
            )

        self.client = Groq(api_key=api_key)
        self.model = model

        self.store = VectorStore()
        self.store.add_documents(knowledge_dir)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.turns: list[dict] = []
        self.total_tokens = 0
        self.session_started = datetime.now()

    def ask(self, query: str, top_n: int, want_quiz: bool) -> dict:
        """Run one full turn: search -> answer -> (optional) quiz question."""
        results = self.store.search(query, top_n=top_n)

        answer = generate_answer(self.client, self.store, query, results, model=self.model)

        quiz_question = None
        if want_quiz and results:
            quiz_question = generate_quiz_question(self.client, self.store, results, model=self.model)

        turn_tokens = count_tokens(query) + count_tokens(answer) + count_tokens(quiz_question or "")
        self.total_tokens += turn_tokens

        turn_record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "user": query,
            "matches": [
                {
                    "source": self.store.get_by_id(idx)["source"],
                    "chunk": self.store.get_by_id(idx)["chunk"],
                    "score": score,
                }
                for idx, score in results
            ],
            "assistant": answer,
            "quiz_question": quiz_question,
            "turn_tokens": turn_tokens,
            "total_tokens_so_far": self.total_tokens,
        }
        self.turns.append(turn_record)
        return turn_record

    def save_log(self, fmt: LogFormat = DEFAULT_LOG_FORMAT) -> Path:
        """Write the full session history to logs/{timestamp}.md or .json."""
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if fmt == LogFormat.JSON:
            path = self.log_dir / f"session_{date_str}.json"
            payload = {
                "session_started": self.session_started.isoformat(timespec="seconds"),
                "model": self.model,
                "total_tokens": self.total_tokens,
                "turns": self.turns,
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            path = self.log_dir / f"session_{date_str}.md"
            lines = [
                f"# Study session log — {date_str}",
                "",
                f"**Model:** `{self.model}`  ",
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
                if turn["quiz_question"]:
                    lines.append("")
                    lines.append(f"**Quick check:** {turn['quiz_question']}")
                lines.append("")
                lines.append(f"_Tokens used: {turn['turn_tokens']} | Total so far: {turn['total_tokens_so_far']}_")
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
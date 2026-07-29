# task_2/constants.py
from enum import Enum

class Role(str, Enum):
    """Ролі повідомлень у форматі Groq/OpenAI chat API."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolName(str, Enum):
    """Назви інструментів, які може викликати модель"""

    CALCULATE = "calculate"
    EXPLAIN = "explain"
    FAKE_LOOKUP = "fake_lookup"
    WIKIPEDIA_SEARCH = "wikipedia_search"


class LogFormat(str, Enum):
    """Формат файлу логів для log_tool_call."""

    TXT = "txt"
    MD = "md"
    JSON = "json"

# task_2/constants.py
from enum import Enum


class Role(str, Enum):
    """Chat message roles used with the Groq API."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolName(str, Enum):
    """Names of tools available to the assistant."""
    CALCULATE = "calculate"
    EXPLAIN = "explain"
    FAKE_LOOKUP = "fake_lookup"
    WIKIPEDIA_SEARCH = "wikipedia_search"


class LogFormat(str, Enum):
    """Supported formats for writing logs to disk."""
    TXT = "txt"
    MD = "md"
    JSON = "json"

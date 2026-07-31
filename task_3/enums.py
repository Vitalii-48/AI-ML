from enum import Enum


class LogFormat(str, Enum):
    """Supported formats for session logs."""
    MD = "md"
    JSON = "json"


class Role(str, Enum):
    """Chat message roles, matching the Groq/OpenAI message schema. """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
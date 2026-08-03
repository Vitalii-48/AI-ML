from enum import Enum


class LogFormat(str, Enum):
    """Supported formats for session logs."""
    MD = "md"
    JSON = "json"

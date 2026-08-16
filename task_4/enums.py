from enum import Enum


class Mode(str, Enum):
    """Supported transcript-processing modes."""
    SUMMARY = "summary"
    EXTRACT_KEYWORDS = "extract_keywords"
    GENERATE_TITLE = "generate_title"
    QNA = "qna"

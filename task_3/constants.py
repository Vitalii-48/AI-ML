from pathlib import Path

from enums import LogFormat

# Absolute path to this script's folder — so "knowledge" and "logs"
SCRIPT_DIR = Path(__file__).resolve().parent

# Local embedding model (sentence-transformers)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Groq model used for answers and quiz questions
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"

# Knowledge base folder
DEFAULT_KNOWLEDGE_DIR = str(SCRIPT_DIR / "knowledge")

# Number of top matches to retrieve by default
DEFAULT_TOP_N = 3

# Session logging
DEFAULT_LOG_DIR = str(SCRIPT_DIR / "logs")
DEFAULT_LOG_FORMAT: LogFormat = LogFormat.MD

# List of extension support
SUPPORTED_FILE_EXTENSIONS = ("*.txt",)
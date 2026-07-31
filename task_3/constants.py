# task_3\constants.py
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
DEFAULT_KNOWLEDGE_DIR = str(SCRIPT_DIR / "knowledge")
DEFAULT_TOP_N = 3
SUPPORTED_FILE_EXTENSIONS = ("*.txt",)
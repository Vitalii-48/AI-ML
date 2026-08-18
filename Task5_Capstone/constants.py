from pathlib import Path

DEFAULT_MODEL_NAME = "llama-3.3-70b-versatile"
WHISPER_MODEL_NAME = "whisper-large-v3"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Absolute path to this script's folder
SCRIPT_DIR = Path(__file__).resolve().parent

AUDIO_DIR = SCRIPT_DIR / "audio"

SESSION_DIR = SCRIPT_DIR / "sessions"

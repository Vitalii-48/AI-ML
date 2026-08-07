from pathlib import Path


# Absolute path to this script's folder
SCRIPT_DIR = Path(__file__).resolve().parent

# Default directory for audio files.
AUDIO_DIR = SCRIPT_DIR / "audio"

# Groq model used for answers and quiz questions
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"

# Whisper model used for audio transcription.
DEFAULT_WHISPER_MODEL = "whisper-large-v3"
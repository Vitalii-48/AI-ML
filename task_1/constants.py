# task_1\constants.py
from enums import LogFormat

MODEL = "llama-3.3-70b-versatile"
SYSTEM_PROMPT = (
    "You are a math tutor. Explain mathematical concepts step by step, "
    "use formulas and examples, ask follow-up questions, and check the student's understanding."
)
DEFAULT_SYSTEM_PROMPT = (
    "You are a patient, knowledgeable tutor. Explain concepts clearly, "
    "ask follow-up questions, and check the user's understanding as you go."
)
DEFAULT_LOG_DIR = "logs"
LOG_FORMAT: LogFormat = LogFormat.MD

# task_2/constants.py
from enums import LogFormat, ToolName

# Model Groq
MODEL_NAME = "llama-3.3-70b-versatile"

# Logging
LOGS_DIR = "logs"
DEFAULT_LOG_FORMAT = LogFormat.TXT  # or LogFormat.MD / LogFormat.JSON
FAKE_DB = "fake_db.json"

# Date/time formats for logs
DATE_FORMAT = "%Y-%m-%d"
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"

USER_AGENT_WIKI = "EducationalCLIAssistant/1.0 (dev_test@example.com)"
ALLOWED_CALC_CHARS = "0123456789+-*/()., "

SYSTEM_PROMPT = (
    f"You are a useful CLI learning assistant. You have access to tools: "
    f"'{ToolName.CALCULATE}', '{ToolName.EXPLAIN}', '{ToolName.FAKE_LOOKUP}', and '{ToolName.WIKIPEDIA_SEARCH}'.\n\n"
    "Follow these formatting rules depending on the situation:\n\n"
    "1. IF A LOCAL FUNCTION IS USED:\n"
    f"- Use '{ToolName.WIKIPEDIA_SEARCH}' ONLY if the user explicitly mentions the word 'Wikipedia' in their request.\n"
    f"- Use '{ToolName.FAKE_LOOKUP}' for general informational queries where Wikipedia is not explicitly mentioned.\n"
    "You must mention the source of information and its literal result. Format your response exactly according to this template:\n"
    "According to the source [function_name], the value is: [exact_tool_result].\n"
    "[Your brief personal comment or addition here, if necessary].\n\n"
    "Note: Replace [function_name] with the actual name of the called tool and [exact_tool_result] with its exact return value word-for-word.\n\n"
    "2. IF A TOOL IS NOT NEEDED:\n"
    "Give your own, independent answer. In this case, do NOT mention any local functions, tools, or sources. The answer must be simple, direct, and concise."
)
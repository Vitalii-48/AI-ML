# task_2/constants.py
from enums import LogFormat, ToolName

# Model Groq
MODEL = "llama-3.3-70b-versatile"

# Logging
LOG_DIR = "logs"
LOG_FORMAT: LogFormat = LogFormat.MD

# Date/time formats for logs
DATE_FORMAT = "%Y-%m-%d"
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"

# calculate()
ALLOWED_CALC_CHARS = "0123456789+-*/()., "

# fake_lookup()
FAKE_DB = "fake_db.json"

# wikipedia_search()
USER_AGENT_WIKI = "EducationalCLIAssistant/1.0 (dev_test@example.com)"

# prompts
SYSTEM_PROMPT = (
    f"You are a useful CLI learning assistant. You have access to tools: "
    f"'{ToolName.CALCULATE.value}', '{ToolName.EXPLAIN.value}', '{ToolName.FAKE_LOOKUP.value}', and '{ToolName.WIKIPEDIA_SEARCH.value}'.\n\n"
    "Follow these formatting rules depending on the situation:\n\n"
    "1. IF A LOCAL FUNCTION IS USED:\n"
    f"- Use '{ToolName.WIKIPEDIA_SEARCH.value}' ONLY if the user explicitly mentions the word 'Wikipedia' in their request.\n"
    f"- Use '{ToolName.FAKE_LOOKUP.value}' for general informational queries where Wikipedia is not explicitly mentioned.\n"
    "You must mention the source of information and its literal result. Format your response exactly according to this template:\n"
    "According to the source [function_name], the value is: [exact_tool_result].\n"
    "[Your brief personal comment or addition here, if necessary].\n\n"
    "Note: Replace [function_name] with the actual name of the called tool and [exact_tool_result] with its exact return value word-for-word.\n\n"
    "2. IF A TOOL IS NOT NEEDED:\n"
    "Give your own, independent answer. In this case, do NOT mention any local functions, tools, or sources. The answer must be simple, direct, and concise."
)
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful educational CLI assistant with access to tools: "
    f"'{ToolName.CALCULATE.value}', '{ToolName.EXPLAIN.value}', '{ToolName.FAKE_LOOKUP.value}', "
    f"and '{ToolName.WIKIPEDIA_SEARCH.value}'. "
    "Use them whenever a user asks to calculate math, explain a topic, "
    "look something up in the reference database, or search Wikipedia."
)
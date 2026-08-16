# task_2\tools.py
import datetime
import json
import os

import wikipediaapi

from constants import (
    ALLOWED_CALC_CHARS,
    DATE_FORMAT,
    LOG_FORMAT,
    FAKE_DB,
    LOG_DIR,
    TIMESTAMP_FORMAT,
    USER_AGENT_WIKI,
)
from enums import LogFormat, ToolName

# Base directory of this file, so paths work regardless of the current
# working directory the script is launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR_PATH = os.path.join(BASE_DIR, LOG_DIR)
FAKE_DB_PATH = os.path.join(BASE_DIR, FAKE_DB)


# =====================================================================
# Logging
# =====================================================================
def log_tool_call(function_name: str, args: dict, fmt: LogFormat = LOG_FORMAT) -> None:
    """Append a record of a tool call to logs/YYYY-MM-DD.<fmt>."""
    os.makedirs(LOGS_DIR_PATH, exist_ok=True)

    timestamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
    current_date = datetime.datetime.now().strftime(DATE_FORMAT)
    file_path = os.path.join(LOGS_DIR_PATH, f"{current_date}.{fmt.value}")

    if fmt == LogFormat.JSON:
        entry = {"timestamp": timestamp, "tool": function_name, "args": args}
        line = json.dumps(entry, ensure_ascii=False) + "\n"
    elif fmt == LogFormat.MD:
        line = f"- **[{timestamp}]** Tool: `{function_name}` | Args: `{json.dumps(args, ensure_ascii=False)}`\n"
    else:
        line = f"[{timestamp}] Tool: {function_name} | Args: {json.dumps(args, ensure_ascii=False)}\n"

    with open(file_path, "a", encoding="utf-8") as f_log:
        f_log.write(line)


# =====================================================================
# Tool functions
# =====================================================================
def calculate(expr: str) -> str:
    """Evaluate a simple math expression using only digits and + - * / ( )."""
    try:
        if not all(char in ALLOWED_CALC_CHARS for char in expr):
            return "Error: expression contains invalid characters."
        result = eval(expr)  # noqa: S307 - input already validated above
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"


def explain(topic: str) -> str:
    """Explain a study topic using a small local knowledge base."""
    knowledge_base = {
        "photosynthesis": "Photosynthesis is the process used by plants to convert light energy into chemical energy.",
        "gravity": "Gravity is a natural phenomenon by which all things with mass or energy are brought toward one another.",
    }
    topic_lower = topic.lower().strip()
    if topic_lower in knowledge_base:
        return knowledge_base[topic_lower]
    return f"Topic '{topic}' was not found in the local knowledge base."


def fake_lookup(query: str) -> str:
    """Simulate access to a reference database by reading a local JSON file."""
    try:
        with open(FAKE_DB_PATH, "r", encoding="utf-8") as f:
            fake_database = json.load(f)
    except FileNotFoundError:
        return f"Error: database file '{FAKE_DB_PATH}' not found."

    query_lower = query.lower().strip()
    if query_lower in fake_database:
        return fake_database[query_lower]
    return f"No entry found for '{query}'."


def wikipedia_search(query: str) -> str:
    """Search live Wikipedia and return the first paragraph of the summary."""
    wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT_WIKI, language="en")
    try:
        page = wiki.page(query.strip())
        if page.exists():
            return page.summary.split("\n")[0]
        return f"Information about '{query}' was not found on Wikipedia."
    except Exception as e:
        return f"Error connecting to Wikipedia: {str(e)}"


# =====================================================================
# Dispatch map: tool name -> Python function
# =====================================================================
TOOL_FUNCTIONS = {
    ToolName.CALCULATE.value: calculate,
    ToolName.EXPLAIN.value: explain,
    ToolName.FAKE_LOOKUP.value: fake_lookup,
    ToolName.WIKIPEDIA_SEARCH.value: wikipedia_search,
}
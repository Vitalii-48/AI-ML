# task_2\tools.py
import datetime
import json
import os
import wikipediaapi

from enums import LogFormat, ToolName

from constants import (
    ALLOWED_CALC_CHARS,
    DATE_FORMAT,
    DEFAULT_LOG_FORMAT,
    FAKE_DB,
    LOGS_DIR,
    TIMESTAMP_FORMAT,
    USER_AGENT_WIKI,
)

# Базова директорія проєкту (там, де лежить цей файл).
# Завдяки цьому шляхи до fake_db.json та logs/ не залежать від того,
# з якої робочої директорії запущено скрипт.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, LOGS_DIR)
FAKE_DB_PATH = os.path.join(BASE_DIR, FAKE_DB)


# =====================================================================
# ЛОГУВАННЯ ВИКЛИКІВ ІНСТРУМЕНТІВ
# =====================================================================
def log_tool_call(function_name: str, args: dict, fmt: LogFormat = DEFAULT_LOG_FORMAT) -> None:
    """Записує факт виклику інструмента у logs/YYYY-MM-DD.<fmt>.

    fmt: LogFormat - у якому форматі писати лог (txt / md / json).
    За замовчуванням DEFAULT_LOG_FORMAT (LogFormat.TXT).
    """
    os.makedirs(LOGS_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
    current_date = datetime.datetime.now().strftime(DATE_FORMAT)
    file_path = os.path.join(LOGS_DIR, f"{current_date}.{fmt.value}")

    if fmt == LogFormat.JSON:
        # JSON Lines: один JSON-об'єкт на рядок — зручно для append-логів,
        # без потреби перечитувати й переписувати весь файл.
        entry = {"timestamp": timestamp, "tool": function_name, "args": args}
        line = json.dumps(entry, ensure_ascii=False) + "\n"
    elif fmt == LogFormat.MD:
        line = f"- **[{timestamp}]** Tool: `{function_name}` | Args: `{json.dumps(args, ensure_ascii=False)}`\n"
    else:  # LogFormat.TXT
        line = f"[{timestamp}] Tool: {function_name} | Args: {json.dumps(args, ensure_ascii=False)}\n"

    with open(file_path, "a", encoding="utf-8") as f_log:
        f_log.write(line)


# =====================================================================
# ЛОКАЛЬНІ PYTHON-ФУНКЦІЇ (ІНСТРУМЕНТИ / TOOLS)
# =====================================================================
def calculate(expr: str) -> str:
    """Обчислює прості математичні вирази"""
    try:
        if not all(char in ALLOWED_CALC_CHARS for char in expr):
            return "Error: Вираз містить недозволені символи."

        result = eval(expr)  # noqa: S307 - вхід уже провалідований ALLOWED_CALC_CHARS
        return str(result)

    except Exception as e:
        return f"Помилка обчислення: {str(e)}"


def explain(topic: str) -> str:
    """Пояснює тему з локальної бази знань"""
    knowledge_base = {
        "photosynthesis": "Photosynthesis is the process used by plants to convert light energy into chemical energy.",
        "gravity": "Gravity is a natural phenomenon by which all things with mass or energy are brought toward one another.",
    }
    topic_lower = topic.lower().strip()
    if topic_lower in knowledge_base:
        return knowledge_base[topic_lower]
    return f"Тему '{topic}' не знайдено в локальній базі. Опиши її самостійно своїми словами."


def fake_lookup(query: str) -> str:
    """Симулює звернення до енциклопедії, читаючи дані з JSON-файлу"""
    try:
        with open(FAKE_DB_PATH, "r", encoding="utf-8") as f:
            fake_database = json.load(f)
    except FileNotFoundError:
        return f"Error: файл бази даних '{FAKE_DB_PATH}' не знайдено."

    query_lower = query.lower().strip()
    if query_lower in fake_database:
        return fake_database[query_lower]
    return f"No entry found for '{query}'."


def wikipedia_search(query: str) -> str:
    """Шукає інформацію на реальній Вікіпедії."""
    wiki = wikipediaapi.Wikipedia(
        user_agent=USER_AGENT_WIKI,
        language="en",
    )
    try:
        page = wiki.page(query.strip())
        if page.exists():
            # Беремо перший абзац
            return page.summary.split("\n")[0]
        return f"Information about '{query}' not found on Wikipedia."
    except Exception as e:
        return f"Error connecting to Wikipedia: {str(e)}"


# =====================================================================
# JSON-ОПИС ІНСТРУМЕНТІВ ДЛЯ ШІ (function calling schema)
# =====================================================================
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": ToolName.CALCULATE.value,
            "description": "Solve math expressions like '5 * (2 + 3)'. Use only numbers and operators +,-,*,/,().",
            "parameters": {
                "type": "object",
                "properties": {
                    "expr": {"type": "string", "description": "The math expression to solve."}
                },
                "required": ["expr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": ToolName.EXPLAIN.value,
            "description": "Explain a study topic in simple terms (e.g., 'photosynthesis', 'gravity').",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The educational topic to explain."}
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": ToolName.FAKE_LOOKUP.value,
            "description": "Look up information in the local reference database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The term or topic to look up, e.g. 'python' or 'einstein'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": ToolName.WIKIPEDIA_SEARCH.value,
            "description": "Search the live Wikipedia API. Use ONLY when the user explicitly asks to search Wikipedia.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search term to look up on Wikipedia."}
                },
                "required": ["query"],
            },
        },
    },
]

# Диспетчеризація: назва інструмента -> Python-функція.
# Ключі беремо з Enum ToolName (а не дублюємо рядки), щоб назви завжди
# лишались синхронізованими зі схемою вище.
TOOL_FUNCTIONS = {
    ToolName.CALCULATE.value: calculate,
    ToolName.EXPLAIN.value: explain,
    ToolName.FAKE_LOOKUP.value: fake_lookup,
    ToolName.WIKIPEDIA_SEARCH.value: wikipedia_search,
}

# task 2\tool_assistant_day2.py
import datetime
import os
import json
import wikipediaapi

from dotenv import load_dotenv
from groq import Groq

# Завантажуємо змінні оточення з файлу .env
load_dotenv()

# Отримуємо API-ключ та ініціалізуємо офіційний клієнт Groq
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("Помилка: GROQ_API_KEY не знайдено в .env файлі!")
client = Groq(api_key=api_key)


def log_tool_call(function_name: str, args: dict) :
    if not os.path.exists("logs"):
        os.makedirs("logs")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    file_path = f"logs/{current_date}.log"

    log_entry = f"[{timestamp}] Tool: {function_name} | Args: {json.dumps(args, ensure_ascii=False)}\n"

    with open(file_path, "a", encoding="utf-8") as f_l:
        f_l.write(log_entry)
# =====================================================================
# 1. ЛОКАЛЬНІ PYTHON-ФУНКЦІЇ (ІНСТРУМЕНТИ / TOOLS)
# =====================================================================
def calculate(expr: str) -> str:
    """Обчислює прості математичні вирази"""
    try:
        allowed_char = "0123456789+-*/(). "
        if not all(char in allowed_char for char in expr):
            return "Error: Вираз містить недозволені символи."

        result = eval(expr)
        return str(result)

    except Exception as e:
        return f" Помилка обчислення: {str(e)}"


def explain(topic: str) -> str:
    """Пояснює тему з локальної бази знань"""
    knowledge_base = {
        "photosynthesis": "Photosynthesis is the process used by plants to convert light energy into chemical energy.",
        "gravity": "Gravity is a natural phenomenon by which all things with mass or energy are brought toward one another."
    }
    topic_lower = topic.lower().strip()
    if topic_lower in knowledge_base:
        return knowledge_base[topic_lower]
    else:
        return f"Тему '{topic}' не знайдено в локальній базі. Опиши її самостійно своїми словами."


def fake_lookup(query: str) -> str:
    """Симулює звернення до енциклопедії, читаючи дані з JSON-файлу"""
    with open("fake_db.json", "r", encoding="utf-8") as f:
        fake_database = json.load(f)

    query_lower = query.lower().strip()
    if query_lower in fake_database:
        return fake_database[query_lower]
    else:
        return f"No entry found for '{query}'."


def wikipedia_search(query: str) -> str:
    """Шукає інформацію на реальній Вікіпедії."""
    wiki = wikipediaapi.Wikipedia(
        user_agent="EducationalCLIAssistant/1.0 (your_email@example.com)",
        language="en"
    )
    try:
        page = wiki.page(query.strip())
        if page.exists():
            # Беремо перший абзац
            return page.summary.split('\n')[0]
        return f"Information about '{query}' not found on Wikipedia."
    except Exception as e:
        return f"Error connecting to Wikipedia: {str(e)}"


# ==========================================
# 2. JSON-ОПИС ІНСТРУМЕНТІВ ДЛЯ ШІ (JSON/DICT)
# ==========================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Solve math expressions like '5 * (2 + 3)'. Use only numbers and operators +,-,*,/,().",
            "parameters": {
                "type": "object",
                "properties": {
                    "expr": {"type": "string", "description": "The math expression to solve."}
                },
                "required": ["expr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explain",
            "description": "Explain a study topic in simple terms (e.g., 'photosynthesis', 'gravity').",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The educational topic to explain."}
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fake_lookup",
            "description": "Look up information in the local reference database. IMPORTANT: Generate valid JSON with a space after the function name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The term or topic to look up, e.g. 'python' or 'einstein'."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Search the live Wikipedia API. Use ONLY when the user explicitly asks to search Wikipedia.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search term to look up on Wikipedia."}
                },
                "required": ["query"]
            }
        }
    }
]


# =====================================================================
# 3. ОСНОВНА ШІ-ЛОГІКА (FUNCTION CALLING CHAT FLOW)
# =====================================================================

def ask_groq(question: str) -> str | None:
    """Відправляє питання в Groq API і повертає повну відповідь (без стрімінгу)."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a useful CLI learning assistant. You have access to tools: 'calculate', 'explain', 'fake_lookup', and 'wikipedia_search'.\n\n"
                "Follow these formatting rules depending on the situation:\n\n"
                "1. IF A LOCAL FUNCTION IS USED:\n"
                "- Use 'wikipedia_search' ONLY if the user explicitly mentions the word 'Wikipedia' in their request.\n"
                "- Use 'fake_lookup' for general informational queries where Wikipedia is not explicitly mentioned.\n"
                "You must mention the source of information and its literal result. Format your response exactly according to this template:\n"
                "According to the source [function_name], the value is: [exact_tool_result].\n"
                "[Your brief personal comment or addition here, if necessary].\n\n"
                "Note: Replace [function_name] with the actual name of the called tool and [exact_tool_result] with its exact return value word-for-word.\n\n"
                "2. IF A TOOL IS NOT NEEDED:\n"
                "Give your own, independent answer. In this case, do NOT mention any local functions, tools, or sources. The answer must be simple, direct, and concise."            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    try:
        # Перший запит до Groq
        response = client.chat.completions.create( # type: ignore
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        response_message = response.choices[0].message

        # Перевіряємо, чи викликає модель інструменти
        if response_message.tool_calls:
            messages.append(response_message) # type: ignore
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                log_tool_call(function_name, function_args)

                if function_name == "calculate":
                    tool_result = calculate(function_args["expr"])
                elif function_name == "explain":
                    tool_result = explain(function_args["topic"])
                elif function_name == "fake_lookup":
                    tool_result = fake_lookup(function_args["query"])
                elif function_name == "wikipedia_search":
                    tool_result = wikipedia_search(function_args["query"])
                else:
                    tool_result = "Error: Unknown tool call."

                messages.append ({
                    "role": "tool",
                     "tool_call_id": tool_call.id,
                     "name": function_name,
                     "content": tool_result
                })

            # Другий запит до Groq з урахуванням результатів функцій
            final_response = client.chat.completions.create(  # type: ignore
                model="llama-3.3-70b-versatile",
                messages=messages,
                stream=False
            )


            return final_response.choices[0].message.content

        else:
            # Звичайне питання користувача
            return response_message.content

    except Exception as e:
        return f"Вибач, сталася помилка при зверненні до асистента: {e}. Спробуй перефразувати запитання."




# =====================================================================
# 4. ГОЛОВНИЙ CLI-ІНТЕРФЕЙС ПРОГРАМИ
# =====================================================================

def main():
    print("Вітаю у навчальному CLI-асистенті")
    print("Введіть ваше запитання або напишіть exit для виходу.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if not user_input:
            continue

        assistent_reply = ask_groq(user_input)
        print(f"Assistent: '{assistent_reply}'\n")

if __name__ == "__main__":
    main()

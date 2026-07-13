# task 2\tool_assistant_day2.py

import os
import json

from dotenv import load_dotenv
from groq import Groq

# Завантажуємо змінні оточення з файлу .env
load_dotenv()

# Отримуємо API-ключ та ініціалізуємо офіційний клієнт Groq
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

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
            "content": "You are a helpful educational CLI assistant. Use tools whenever a user asks to calculate math or explain topics."
        },
        {
            "role": "user",
            "content": question
        }
    ]
    response = client.chat.completions.create( # type: ignore
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        stream=False,
    )

    response_message = response.choices[0].message

    if response_message.tool_calls:
        messages.append(response_message) # type: ignore
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            if function_name == "calculate":
                tool_result = calculate(function_args["expr"])
            elif function_name == "explain":
                tool_result = explain(function_args["topic"])
            else:
                tool_result = "Error: Unknown tool call."

            messages.append ({
                "role": "tool",
                 "tool_call_id": tool_call.id,
                 "name": function_name,
                 "content": tool_result
            })

        final_response = client.chat.completions.create(  # type: ignore
            model="llama-3.3-70b-versatile",
            messages=messages,
            stream=False
        )


        return final_response.choices[0].message.content

    else:
        return  response_message.content


# =====================================================================
# 4. ГОЛОВНИЙ CLI-ІНТЕРФЕЙС ПРОГРАМИ
# =====================================================================

def main():
    print("Вітаю у навчальному CLI-асистенті")
    print("Введіть ваше запитання або напишвть exit для виходу.\n")

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

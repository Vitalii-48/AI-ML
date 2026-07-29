# task 2\tool_assistant_day2.py
import os
import json

from dotenv import load_dotenv
from groq import Groq

from constants import MODEL_NAME, SYSTEM_PROMPT
from enums import Role
from tools import TOOLS_SCHEMA, TOOL_FUNCTIONS, log_tool_call

# Завантажуємо змінні оточення з файлу .env
load_dotenv()

# Отримуємо API-ключ та ініціалізуємо офіційний клієнт Groq
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Create a .env file with "
        "GROQ_API_KEY=<your key> next to this script."
    )
client = Groq(api_key=api_key)

# =====================================================================
# ОСНОВНА ШІ-ЛОГІКА (FUNCTION CALLING CHAT FLOW)
# =====================================================================

def ask_groq(messages: list) -> str | None:
    """
    Приймає та оновлює повний масив повідомлень (історію),
    відправляє запити до Groq API і повертає текст відповіді.
    """

    try:
        # Перший запит до Groq
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            stream=False,
        )

        response_message = response.choices[0].message

        # Перевіряємо, чи викликає модель інструменти
        if response_message.tool_calls:
            messages.append(response_message.model_dump())
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                log_tool_call(function_name, function_args)
                tool_func = TOOL_FUNCTIONS.get(function_name)

                if tool_func:
                    # Отримуємо перший аргумент зі словника args (напр. expr, topic або query)
                    arg_value = next(iter(function_args.values())) if function_args else ""
                    tool_result = tool_func(arg_value)
                else:
                    tool_result = f"Error: Unknown tool call '{function_name}'."

                messages.append({
                    "role": Role.TOOL.value,
                     "tool_call_id": tool_call.id,
                     "name": function_name,
                     "content": tool_result
                })

            # Другий запит до Groq з урахуванням результатів функцій
            final_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=False
            )
            final_content = final_response.choices[0].message.content or ""
            messages.append(
                {"role": Role.ASSISTANT.value, "content": final_content}
            )
            return final_content

        else:
            content = response_message.content or ""
            messages.append({
                "role": Role.ASSISTANT.value,
                "content": content
            })

            return content

    except Exception as e:
        return f"Вибач, сталася помилка при зверненні до асистента: {e}. Спробуй перефразувати запитання."


# =====================================================================
# ГОЛОВНИЙ CLI-ІНТЕРФЕЙС ПРОГРАМИ
# =====================================================================

def main():
    print("Вітаю у навчальному CLI-асистенті")
    print("Введіть ваше запитання або напишіть exit для виходу.\n")

    messages = [{"role": Role.SYSTEM.value, "content": SYSTEM_PROMPT}]

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if not user_input:
            continue

        messages.append({"role": Role.USER.value, "content": user_input})

        assistant_reply = ask_groq(messages)
        print(f"Assistant: '{assistant_reply}'\n")

if __name__ == "__main__":
    main()

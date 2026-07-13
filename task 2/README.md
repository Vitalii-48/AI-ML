# Task 2: Educational CLI Assistant with Function Calling

This is a CLI study assistant powered by the **Groq SDK** (using the `llama-3.3-70b-versatile` model). It uses **Function Calling** to detect user intents and delegate tasks to local Python functions.

## Features & Tools
* **`calculate(expr)`**: A safe math calculator. It uses a "white list" of allowed characters (`0123456789+-*/(). `) to evaluate math expressions securely.
* **`explain(topic)`**: A local knowledge base that looks up terms (like `photosynthesis` or `gravity`) from a dictionary and provides accurate facts to the LLM.

## How It Works (Chat Flow)
1. **First Request**: The user asks a question. The app sends the question along with the `tools` description to Groq.
2. **Tool Trigger**: The model detects that a function is needed and returns a `tool_calls` payload.
3. **Local Execution**: The Python script executes the local function, gets the result, and appends it to the chat history.
4. **Second Request**: The app sends the updated history back to Groq, and the model generates a final response for the user.

## Sample Run (Terminal Output)
```text
Вітаю у навчальному CLI-асистенті
Введіть ваше запитання або напишіть exit для виходу.

You: Calculate 125 * (4 + 6)
Assistent: 'The result of the calculation 125 * (4 + 6) is 1250. This is calculated by first adding 4 and 6, which equals 10, and then multiplying 125 by 10.'

You: Скільки буде 45 * 2 - 10?
Assistent: 'Відповідь: 80'

You: Explain photosynthesis
Assistent: 'Photosynthesis is the process by which plants, algae, and some bacteria convert light energy from the sun into chemical energy...'
```

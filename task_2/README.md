# Task 2: Educational CLI Assistant with Function Calling

This is a CLI study assistant powered by the **Groq SDK** (using the `llama-3.3-70b-versatile` model). It uses **Function Calling** to detect user intents and delegate tasks to local Python functions.

## Features & Tools
* **`calculate(expr)`**: A safe math calculator. It uses a "white list" of allowed characters (`0123456789+-*/(). `) to evaluate math expressions securely.
* **`explain(topic)`**: A local knowledge base that looks up terms (like `photosynthesis` or `gravity`) from a dictionary and provides accurate facts to the LLM.
* **`fake_lookup(query)`**: Simulates an internal corporate/educational reference database by reading data dynamically from a local `fake_db.json` file.
* **`wikipedia_search(query)`**: A live internet tool integrated via the `wikipedia-api` library. It fetches real-time summaries directly from Wikipedia, triggered automatically whenever the user explicitly mentions "Wikipedia".

## How It Works (Chat Flow)
1. **First Request**: The user asks a question. The app sends the user input along with JSON schemas of the `tools` to Groq.
2. **Intent & Routing Detection**: The model evaluates the query based on strict system prompt instructions:
   * Math queries go to `calculate`.
   * General info queries check the local `fake_lookup` database first.
   * Direct Wikipedia requests route straight to the live `wikipedia_search` API.
3. **Local Execution**: The Python script intercepts the `tool_calls` payload, executes the respective function, logs the action inside the `/logs` directory, and appends the raw result back to the chat messages.
4. **Final Response**: The app sends the updated history back to Groq, and the model synthesizes a clean, user-friendly response citing the exact tool source used.

## Sample Run (Terminal Output)
```text
Вітаю у навчальному CLI-асистенті
Введіть ваше запитання або напишіть exit для виходу.

You: Calculate 125 * (4 + 6)
Assistent: 'According to the source calculate, the value is: 1250. This is calculated by first adding 4 and 6, which equals 10, and then multiplying 125 by 10.'

You: What Wikipedia says about bananas
Assistent: 'According to the source wikipedia_search, the value is: A banana is an elongated, edible fruit that is botanically a berry produced by several kinds of large treelike herbaceous flowering plants...'

You: tell me about gravity and Einstein
Assistent: 'According to the source fake_lookup, the value is: No entry found for 'gravity and Einstein'.
Albert Einstein is famous for his theory of general relativity, which revolutionized our understanding of gravity...'
```

# Task 2 — Educational CLI Assistant with Function Calling

A command-line educational assistant powered by the **Groq API**. The application
extends the chat assistant from Task 1 by adding **Function Calling**, allowing
the model to invoke local Python tools while maintaining a multi-turn
conversation history.

## File structure

```text
task_2/
├── app.py                # CLI entry point
├── chat_session.py       # conversation management and Groq API interaction
├── constants.py          # application configuration
├── tools.py              # local tool implementations
├── tool_schemas.py       # JSON schemas for Function Calling
├── fake_db.json          # sample local knowledge base
├── logs/                 # conversation and tool logs
└── README.md
```

## Setup

1. Install project dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file and add your Groq API key:

```text
GROQ_API_KEY=your_groq_api_key_here
```

## Running the application

```bash
python app.py
```

## Available tools

- **calculate(expr)** — safely evaluates arithmetic expressions.
- **explain(topic)** — returns information from a built-in knowledge base.
- **fake_lookup(query)** — searches a local JSON database.
- **wikipedia_search(query)** — retrieves live summaries from Wikipedia.

## Conversation flow

1. The user sends a message.
2. The conversation history and available tool schemas are sent to the Groq API.
3. If the model requests a tool, the corresponding Python function is executed.
4. The tool result is returned to the model.
5. The assistant generates a natural language response.
6. The conversation history is updated after a successful response.

## Logging

The application stores conversation logs during the session. Log format is
configured through the application constants.

## Error handling

The application handles common runtime errors, including:

- missing API key;
- API errors;
- invalid tool execution;
- connection and timeout errors.

## Core Features

- ✅ Multi-turn conversations
- ✅ Function Calling via the Groq API
- ✅ Automatic tool selection
- ✅ Local Python tool execution
- ✅ Conversation history

## Bonus Features

- ✅ Local JSON knowledge base (`fake_lookup`)
- ✅ Live Wikipedia integration
- ✅ Conversation logging
- ✅ Configurable application constants
- ✅ Safe mathematical expression evaluation
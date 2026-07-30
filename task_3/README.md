# Task 3 — Semantic Search Assistant (RAG)

A command-line study assistant powered by the **Groq API** and local
**sentence-transformers embeddings**. The application implements a
Retrieval-Augmented Generation (RAG) pipeline: it embeds a multi-topic
knowledge base, retrieves the most relevant passages for a user's question
via cosine similarity, and asks an LLM to answer using only that retrieved
context.

## File structure

```text
task_3/
├── semantic_search_day3.py   # CLI entry point (argparse + main loop)
├── constants.py               # application configuration
├── models.py                  # Document schema (TypedDict)
├── vector_store.py            # VectorStore class — loading, embedding, search
├── llm.py                     # Groq client, answer/quiz generation
├── knowledge/                 # .txt knowledge base (biology, countries, economics, python)
├── requirements.txt
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
python semantic_search_day3.py
```

### CLI options

| Flag | Description | Default |
| --- | --- | --- |
| `--knowledge` | Path to the knowledge base folder | `knowledge/` next to the script |
| `--top-n` | Number of top matches to retrieve per question | `3` |
| `--model` | Groq model used for answers and quiz questions | `llama-3.3-70b-versatile` |
| `--no-quiz` | Disable Study Mode (skip the quiz question) | off |

Example:

```bash
python semantic_search_day3.py --top-n 5 --no-quiz
```

## Knowledge base

The `knowledge/` folder contains plain `.txt` files. Each file is split into
paragraphs (separated by a blank line), and each paragraph becomes a
separately embedded, separately searchable chunk. The current knowledge base
covers four topics: **biology**, **countries**, **economics**, and **python**.

## Conversation flow

1. The user types a question.
2. The question is embedded locally (`sentence-transformers`,
   `all-MiniLM-L6-v2`) and compared against the knowledge base via cosine
   similarity.
3. The top-N most relevant paragraphs are shown, with their source file and
   paragraph number.
4. The retrieved passages are sent to Groq as context; the model answers
   using **only** that context, citing sources (e.g. *"According to python
   (paragraph 4)..."*), and honestly says when the context isn't enough.
5. **Study Mode:** a short quiz question is generated from the single best
   match, to check understanding of what was just covered.

## Core Features

- ✅ Local semantic search (no external vector DB required)
- ✅ Retrieval-Augmented Generation via the Groq API
- ✅ Multi-topic knowledge base with automatic file chunking
- ✅ Source highlighting (file + paragraph number)
- ✅ Interactive RAG chat loop

## Bonus Features

- ✅ `VectorStore` class (`add_documents`, `search`, `get_by_id`)
- ✅ Study Mode — follow-up quiz question after each answer
- ✅ Configurable CLI flags (`--knowledge`, `--top-n`, `--model`, `--no-quiz`)
- ✅ Centralized configuration (`constants.py`)
- ✅ Typed message params for the Groq SDK (no `# type: ignore`)

## Error handling

The application handles common runtime errors, including:

- missing API key;
- missing/empty knowledge base folder (created automatically with a warning);
- Groq API errors (caught and reported without crashing the session).
# semantic_search_day3.py

import os

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from pathlib import Path


def load_documents(folder: str) -> list[dict]:
    """Load all text files from the knowledge folder."""
    folder = Path(folder)

    documents = []

    for file_path in folder.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        paragraphs = text.split("\n\n")

        for paragraph_number, paragraph in enumerate(paragraphs, start=1):
            paragraph = paragraph.strip()

            if paragraph:
                documents.append(
                    {
                    "text": paragraph,
                    "source": file_path.stem,
                    "chunk": paragraph_number
                    }
                )

    return documents


corpus = load_documents("knowledge")

model = SentenceTransformer("all-MiniLM-L6-v2")

corpus_embeddings = model.encode([doc["text"] for doc in corpus])


def cosine_similarity(a, b) -> float:
    """Рахує косинусну схожість між двома векторами."""
    return float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    )


def search(query, top_n=3):
    """
    Приймає питання користувача, повертає top_n
    найбільш схожих абзаців з корпусу.
    """
    query_embedding = model.encode(query)

    similarities = []
    for i, doc_embedding in enumerate(corpus_embeddings):
        score = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((i, score))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


load_dotenv()

# Отримуємо API-ключ та ініціалізуємо офіційний клієнт Groq
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Create a .env file with "
        "GROQ_API_KEY=<your key> next to this script."
    )
client = Groq(api_key=api_key)


def generate_answer(query, search_results):
    """
    Приймає питання і результати пошуку, формує контекст
     і питає Groq дати фінальну відповідь.
    """
    context_parts = []
    for rank, (idx, score) in enumerate(search_results, start=1):
        doc = corpus[idx]
        context_parts.append(
            f"Source {rank}\n"
            f"File: {doc['source']}\n"
            f"Paragraph: {doc['chunk']}\n"
            f"{doc['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful study assistant.

Answer the question using ONLY the provided context.

If the answer is not in the context, say:
"I don't have enough information in the provided context."

When possible, mention which source(s) you used in your answer.
For example:
"According to python.txt (paragraph 4)..."

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[                              # type: ignore
            {"role": "user", "content": prompt}
        ],
    )

    return response.choices[0].message.content


def main():
    print("=" * 60)
    print("Study Assistant")
    print("=" * 60)
    print("Hi! I'm your study assistant.")
    print("Ask me anything about Python, biology, economics,")
    print("or world countries, and I'll answer using my knowledge base.")
    print("\n[INFO] Type 'exit' or 'quit' to end the session.\n")
    while True:
        query = input("> You: ").strip()

        if query.lower() in ("exit", "quit"):
            print("\nGoodbye!")
            break

        if not query:
            print("Please enter a question.\n")
            continue

        results = search(query, top_n=3)

        print("\n-> Top matches:")
        for rank, (idx, score) in enumerate(results, start=1):
            doc = corpus[idx]
            print(f"{rank}. [{doc['source'].title()} | paragraph {doc['chunk']}] (score={score:.3f}) {doc['text'][:80]}...")

        answer = generate_answer(query, results)
        print(f"\n-> GPT says:\n{answer}")
        print()

if __name__ == "__main__":
    main()
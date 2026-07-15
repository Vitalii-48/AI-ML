# semantic_search_day3.py

import os
from pathlib import Path
from typing import TypedDict

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


class Document(TypedDict):
    text: str
    source: str
    chunk: int




class VectorStore:
    """Stores documents and their embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents: list[Document] = []
        self.embeddings: np.ndarray | None = None

    @staticmethod
    def load_documents(folder: str) -> list[Document]:
        """Load all text files from the knowledge folder."""
        folder = Path(folder)
        documents: list[Document] = []

        if not folder.exists():
            print(f"[WARNING] Folder '{folder}' does not exist. Creating it.")
            folder.mkdir(parents=True, exist_ok=True)
            return documents

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

    def add_documents(self, folder: str) -> None:
        """Load documents and create normalized embeddings."""
        self.documents = self.load_documents(folder)

        embeddings = self.model.encode(
            [doc["text"] for doc in self.documents]
        )

        # Normalize embeddings once
        self.embeddings = embeddings / np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

    def search(
            self,
            query: str,
            top_n: int = 3,
    ) -> list[tuple[int, float]]:
        """Search for the most relevant documents using cosine similarity."""
        if self.embeddings is None or not self.documents:
            return []

        # Create and normalize query embedding
        query_embedding = self.model.encode(query)
        query_embedding /= np.linalg.norm(query_embedding)

        # Compute cosine similarity
        scores = np.dot(self.embeddings, query_embedding)

        # Get indices of top results
        top_indices = np.argsort(scores)[::-1][:top_n]

        return [
            (int(idx), float(scores[idx]))
            for idx in top_indices
        ]

    def get_by_id(self, doc_id: int) -> Document:
        """Return a document by its index."""
        return self.documents[doc_id]


load_dotenv()

# Get the API key and initialize the official Groq client
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Create a .env file with "
        "GROQ_API_KEY=<your key> next to this script."
    )
client = Groq(api_key=api_key)


def bild_context(store: VectorStore, search_results: list[tuple[int, float]]) -> str:
    """Generates textual context from top search results — with source numbering."""
    context_parts: list[str] = []
    for rank, (idx, _) in enumerate(search_results, start=1):
        doc = store.get_by_id(idx)
        context_parts.append(
            f"Source {rank}\n"
            f"File: {doc['source']}\n"
            f"Paragraph: {doc['chunk']}\n"
            f"{doc['text']}"
        )
    return "\n\n".join(context_parts)


def generate_answer(
    store: VectorStore,
    query: str,
    search_results: list[tuple[int, float]],
) -> str | None:
    """Generate an answer using retrieved documents."""
    context = bild_context(store, search_results)

    prompt = f"""
You are a helpful study assistant.

Answer the question using ONLY the provided context.

If the answer is not in the context, say:
"I don't have enough information in the provided context."

When possible, mention which source(s) you used in your answer.
For example:
"According to python (paragraph 4)..."

Context:
{context}

Question:
{query}

Answer:
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[                              # type: ignore
                {"role": "user", "content": prompt}
            ],
        )
    except Exception as e:
        return f"Error while contacting LLM: {e}"

    return response.choices[0].message.content


def generate_quiz_question(
    store: VectorStore,
    search_results: list[tuple[int, float]],
) -> str | None:
    """Generate one short quiz question based on the retrieved context (Study Mode)."""
    context = bild_context(store, search_results[:1])   # to send a question

    prompt = f"""
You are a study assistant helping the user practice what they just learned.
 
Based ONLY on the context below, write exactly ONE short quiz question
that checks understanding of the material. Do not answer it yourself.
 
Context:
{context}
 
Quiz question:
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[ {"role": "user", "content": prompt}] # type: ignore
        )
    except Exception as e:
        return f"Error while generating quiz question: {e}"

    return response.choices[0].message.content


def main():
    store = VectorStore()
    store.add_documents("knowledge")

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

        results = store.search(query, top_n=3)

        print("\n-> Top matches:")
        for rank, (idx, score) in enumerate(results, start=1):
            doc = store.get_by_id(idx)
            print(
                f"{rank}. [{doc['source']} | Paragraph {doc['chunk']}] "
                f"(score={score:.3f}) {doc['text'][:80]}..."
            )
        answer = generate_answer(store, query, results)
        print(f"\n-> GPT says:\n{answer}")

        quiz_question = generate_quiz_question(store, results)
        print(f"\nQuick check:\n{quiz_question}")
        print()

if __name__ == "__main__":
    main()
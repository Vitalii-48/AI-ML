# task_3\semantic_search_day3.py
import argparse

from constants import DEFAULT_KNOWLEDGE_DIR, DEFAULT_LLM_MODEL, DEFAULT_TOP_N
from llm import generate_answer, generate_quiz_question
from vector_store import VectorStore


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Study Assistant — semantic search + RAG")

    parser.add_argument(
        "--knowledge",
        type=str,
        default=DEFAULT_KNOWLEDGE_DIR,
        help=f"Шлях до папки з базою знань (default: {DEFAULT_KNOWLEDGE_DIR})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Скільки топ-збігів шукати для кожного питання (default: {DEFAULT_TOP_N})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f"Назва Groq-моделі для генерації відповідей (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--no-quiz",
        action="store_true",
        help="Вимкнути Study Mode (не показувати quiz-питання після відповіді)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    store = VectorStore()
    store.add_documents(args.knowledge)

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

        results = store.search(query, top_n=args.top_n)

        print("\n-> Top matches:")
        for rank, (idx, score) in enumerate(results, start=1):
            doc = store.get_by_id(idx)
            print(
                f"{rank}. [{doc['source']} | Paragraph {doc['chunk']}] "
                f"(score={score:.3f}) {doc['text'][:80]}..."
            )

        answer = generate_answer(store, query, results, model=args.model)
        print(f"\n-> GPT says:\n{answer}")

        if not args.no_quiz:
            quiz_question = generate_quiz_question(store, results, model=args.model)
            print(f"\n Quick check:\n{quiz_question}")
        print()


if __name__ == "__main__":
    main()
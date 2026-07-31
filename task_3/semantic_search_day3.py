# task_3\semantic_search_day3.py
import argparse
import sys

from rich.console import Console

from constants import DEFAULT_KNOWLEDGE_DIR, DEFAULT_LLM_MODEL, DEFAULT_TOP_N
from llm import generate_answer, generate_quiz_question
from vector_store import VectorStore


console = Console(force_terminal=True)

def cprint(text: str, style: str = "") -> None:
    console.print(text, style=style)


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Study Assistant — semantic search + RAG")

    parser.add_argument(
        "--knowledge",
        type=str,
        default=DEFAULT_KNOWLEDGE_DIR,
        help=f"Path to the knowledge base folder (default: {DEFAULT_KNOWLEDGE_DIR})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"How many top matches to search for each question (default: {DEFAULT_TOP_N})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f"Name of the Groq model for generating responses (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--no-quiz",
        action="store_true",
        help="Disable Study Mode (do not show quiz questions after answering)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        store = VectorStore()
        store.add_documents(args.knowledge)
    except EnvironmentError as exc:
        cprint(f"[error] {exc}", style="bold red")
        sys.exit(1)


    cprint("=" * 60)
    cprint("Study Assistant")
    cprint("=" * 60)
    cprint("Hi! I'm your study assistant.")
    cprint("Ask me anything about Python, biology, economics,")
    cprint("or world countries, and I'll answer using my knowledge base.")
    cprint("\n[INFO] Type 'exit' or 'quit' to end the session.\n")

    while True:
        query = input("> You: ").strip()

        if query.lower() in ("exit", "quit"):
            cprint("\nGoodbye!", style="bold green")
            break

        if not query:
            cprint("Please enter a question.\n")
            continue

        results = store.search(query, top_n=args.top_n)

        cprint("\n-> Top matches:")
        for rank, (idx, score) in enumerate(results, start=1):
            doc = store.get_by_id(idx)
            cprint(
                f"{rank}. [{doc['source']} | Paragraph {doc['chunk']}] "
                f"(score={score:.3f}) {doc['text'][:80]}..."
            )

        answer = generate_answer(store, query, results, model=args.model)
        cprint(f"\n-> GPT says:\n{answer}")

        if not args.no_quiz:
            quiz_question = generate_quiz_question(store, results, model=args.model)
            cprint(f"\n Quick check:\n{quiz_question}", style="bold blue")
        print()


if __name__ == "__main__":
    main()
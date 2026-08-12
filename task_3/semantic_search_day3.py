import argparse
import sys

from constants import (
    DEFAULT_KNOWLEDGE_DIR,
    DEFAULT_LLM_MODEL,
    DEFAULT_LOG_FORMAT,
    DEFAULT_TOP_N,
)
from enums import LogFormat
from chat_session import StudySession


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
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
        help=f"Number of top matches to retrieve per question (default: {DEFAULT_TOP_N})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f"Groq model used for answers and quiz questions (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--no-quiz",
        action="store_true",
        help="Disable Study Mode (skip the quiz question after each answer)",
    )
    parser.add_argument(
        "--log-format",
        type=str,
        choices=[fmt.value for fmt in LogFormat],
        default=DEFAULT_LOG_FORMAT.value,
        help=f"Session log format (default: {DEFAULT_LOG_FORMAT.value})",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        session = StudySession(knowledge_dir=args.knowledge, model=args.model)
    except EnvironmentError as exc:
        print(f"[error] {exc}")
        sys.exit(1)

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

        try:
            turn = session.ask(query, top_n=args.top_n, want_quiz=not args.no_quiz)
        except RuntimeError as exc:
            print(f"\n[error] {exc}\n")
            continue

        print("\n-> Top matches:")
        for rank, match in enumerate(turn["matches"], start=1):
            print(
                f"{rank}. [{match['source']} | Paragraph {match['chunk']}] "
                f"(score={match['score']:.3f})"
            )

        print(f"\n-> GPT says:\n{turn['assistant']}")

        if turn["quiz_question"]:
            print(f"\nQuick check:\n{turn['quiz_question']}")
        print()

    log_path = session.save_log(fmt=LogFormat(args.log_format))
    print(f"[Session saved to {log_path}]")
    print(session.summary())


if __name__ == "__main__":
    main()
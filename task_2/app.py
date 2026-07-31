# task_2/app.py
"""CLI entry point for the Task 2 educational assistant."""

from dotenv import load_dotenv

from chat_session import ChatSession
from constants import SYSTEM_PROMPT


def main() -> None:
    """Run the CLI chat loop."""
    load_dotenv()

    try:
        session = ChatSession(system_prompt=SYSTEM_PROMPT)
    except EnvironmentError as exc:
        print(f"Error: {exc}")
        return

    print("Welcome to the educational CLI assistant.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        print("Assistant: ", end="", flush=True)

        def on_chunk(text: str | None) -> None:
            print(text, end="", flush=True)

        try:
            turn = session.send_message(user_input, on_chunk=on_chunk)
        except RuntimeError as exc:
            print(f"\nError: {exc}")
            continue

        print(f"\n[Tokens used: {turn['turn_tokens']} | Total so far: {turn['total_tokens_so_far']}]\n")

    session.save_log()
    print(session.summary())


if __name__ == "__main__":
    main()
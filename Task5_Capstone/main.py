import argparse

from chat_session import ChatSession
from constants import DEFAULT_MODEL_NAME, SESSION_DIR

SESSION_DIR.mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="CLI AI Assistant")
    parser.add_argument(
        "-model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Groq model to use (default: llama-3.3-70b-versatile)"
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable voice output using local TTS"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    session = ChatSession(model_name=args.model, voice_enabled=args.voice)

    print(f"Assistant is ready! Using model: {session.model_name}")
    print("Commands: "
          "/update_kb_text, "
          "/update_kb_voice, "
          "/search, "
          "/summarize_session, "
          "/change_prompt, "
          "/save_session, "
          "/load_session, "
          "/exit")

    while True:
        user_input = input("\n> You: ").strip()

        if not user_input:
            continue

        if user_input == "/exit":
            print("Exiting...")
            break

        session.handle_command(user_input)


if __name__ == "__main__":
    main()
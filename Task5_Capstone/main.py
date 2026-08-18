import argparse
from pathlib import Path

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
    return parser.parse_args()


def main():
    args = parse_args()
    session = ChatSession(model_name=args.model)

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

        if user_input == "/exit":
            print("Exiting...")
            break

        elif user_input == "/change_prompt":
            new_prompt_input = input("Enter new prompt or a filename: ").strip()
            if new_prompt_input:
                new_prompt = session.change_prompt(new_prompt_input)
                print(f'Assistant: System prompt updated.\nNew prompt: "{new_prompt}"')
            else:
                print("Assistant: Please provide a new prompt or a filename.")

        elif user_input == "/update_kb_text":
            fact = input("Enter your fact: ")
            session.update_kb_text(fact)
            print(f"[Saved] Fact saved. Total facts in KB: {len(session.knowledge_base)}")

        elif user_input == "/update_kb_voice":
            file_input = input("Enter audio file path (e.g. test_audio1.mp3): ").strip()
            try:
                text = session.update_kb_voice(Path(file_input))
                print(f"[Transcribed] \"{text}\"")
                print(f"[Saved] Total facts in KB: {len(session.knowledge_base)}")
            except (FileNotFoundError, ValueError) as e:
                print(f"Assistant: {e}")

        elif user_input == "/search":
            query = input("Enter your search query: ").strip()
            session.call_tool_forced("semantic_search", query=query)

        elif user_input == "/summarize_session":
            session.call_tool_forced("summarize_session")

        elif user_input == "/save_session":
            filename = session.save_session()
            print(f"Assistant: Session saved to {filename}")

        elif user_input == "/load_session":
            filename = input("Enter filename to load session: ").strip()
            try:
                session.load_session(filename)
                print(f"Assistant: Session loaded from {filename}")
                print(f"[Info] {len(session.messages)} messages, {len(session.knowledge_base)} facts in KB")
            except FileNotFoundError as e:
                print(f"Assistant: {e}")

        else:
            session.chat(user_input)


if __name__ == "__main__":
    main()

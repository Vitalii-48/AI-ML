# task_1/app.py

import sys

from dotenv import load_dotenv
from chat_session import ChatSession

try:
    from rich.console import Console

    console = Console()
    _RICH = True
except ImportError:  # rich is optional, plain print fallback
    console = None
    _RICH = False

# --- Simple settings (edit these directly instead of passing CLI args) ---
MODEL = "llama-3.3-70b-versatile"
SYSTEM_PROMPT = None  # None -> uses ChatSession's default tutor prompt
LOG_FORMAT = "md"  # "md" or "json"


def cprint(text: str, style: str = "") -> None:
    if _RICH:
        console.print(text, style=style)
    else:
        print(text)




def main() -> None:
    load_dotenv()

    try:
        session = ChatSession(model=MODEL, system_prompt=SYSTEM_PROMPT)
    except EnvironmentError as exc:
        cprint(f"[error] {exc}", style="bold red")
        sys.exit(1)

    cprint(f"> System Prompt: {session.system_prompt}", style="italic cyan")
    cprint("Type 'quit' or 'exit' to end the session.\n", style="dim")

    while True:
        try:
            user_input = input("> You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            break

        cprint("[Assistant is typing...]", style="dim italic")

        printed_header = False

        def on_chunk(text: str) -> None:
            nonlocal printed_header
            if not printed_header:
                print("Assistant: ", end="", flush=True)
                printed_header = True
            print(text, end="", flush=True)

        try:
            turn = session.send_message(user_input, on_chunk=on_chunk)
        except RuntimeError as exc:
            cprint(f"\n[error] {exc}", style="bold red")
            continue

        print()  # newline after streamed reply
        cprint(
            f"[Tokens used: {turn['turn_tokens']} | Total so far: {turn['total_tokens_so_far']}]\n",
            style="yellow",
        )

    log_path = session.save_log(fmt=LOG_FORMAT)
    cprint(f"[Conversation saved to {log_path}]", style="bold green")
    cprint(session.summary(), style="bold cyan")


if __name__ == "__main__":
    main()
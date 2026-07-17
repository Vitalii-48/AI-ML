# Task5_Capstone\main.py
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from vector_store import VectorStore
from tools import search_kb, summarize_session

load_dotenv()

knowledge_base = VectorStore()

SCRIPT_DIR = Path(__file__).resolve().parent
AUDIO_DIR = SCRIPT_DIR / "audio"
SESSION_DIR = SCRIPT_DIR / "sessions"
SESSION_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to the user's personal knowledge base "
    "and tools that help answer questions.\n\n"
    "Use the semantic_search tool ONLY when the user asks about information that "
    "may already exist in their knowledge base, such as their name, preferences, "
    "location, possessions, or other personal facts. "
    "Do NOT use semantic_search when the user is simply sharing new information "
    "or making casual conversation.\n\n"
    "Use the summarize_session tool when the user explicitly asks for a "
    "summary or recap of the current conversation (for example: 'summarize', "
    "'recap our chat', or 'what have we talked about?'). "
    "If no tool is appropriate, answer normally using your general knowledge."
)

MODEL_NAME = "llama-3.3-70b-versatile"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Create a .env file with "
        "GROQ_API_KEY=<your key> next to this script."
    )

client = Groq(api_key=api_key)

tools = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Searches the user's knowledge base and returns the most relevant fact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search question"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_session",
            "description": "Summarizes the current chat session into a clean bullet-point summary.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


def parse_args():
    parser = argparse.ArgumentParser(description="CLI AI Assistant")
    parser.add_argument(
        "-model",
        type=str,
        default="llama-3.3-70b-versatile",
        help="Groq model to use (default: llama-3.3-70b-versatile)"
    )
    return parser.parse_args()


def change_prompt(new_prompt_input: str):
    """Update the system prompt from text or a file."""
    if not new_prompt_input:
        print("Assistant: Please provide a new prompt or a filename.")
        return

    file_path = Path(new_prompt_input)

    if file_path.exists() and file_path.is_file():
        new_prompt = file_path.read_text(encoding="utf-8").strip()
        print(f"[Loaded prompt from file: {file_path.name}]")
    else:
        new_prompt = new_prompt_input

    messages[0] = {"role": "system", "content": new_prompt}
    print(f"Assistant: System prompt updated.\nNew prompt: \"{new_prompt}\"")


def update_kb_text():
    """Read a fact from the user and add it to the knowledge base."""
    fact = input("Enter your fact: ")
    knowledge_base.add(fact)
    print(f"[Saved] Fact saved. Total facts in KB: {len(knowledge_base.texts)}")


def update_kb_voice():
    """Transcribe an audio file with Whisper and store the result in the knowledge base."""
    file_input = input("Enter audio file path (e.g. test_audio1.mp3): ").strip()

    file_path = Path(file_input)

    if not file_path.is_absolute() and not file_path.exists():
        file_path = AUDIO_DIR / file_input

    if not file_path.exists():
        print(f"Assistant: File not found: {file_path}")
        return

    try:
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
            )
    except Exception as e:
        print(f"Assistant: Sorry, I couldn't transcribe that file. (Error: {e})")
        return

    text = transcription.text.strip()

    if not text:
        print("Assistant: Transcription was empty, nothing to add.")
        return

    knowledge_base.add(text, metadata={"source": file_path.name})
    print(f"[Transcribed] \"{text}\"")
    print(f"[Saved] Fact saved from voice (source: {file_path.name}). Total facts in KB: {len(knowledge_base)}")


def get_completion(force_tool: str | None= None):
    """Generate a model response and execute tool calls when needed."""
    tool_choice = "auto"
    if force_tool:
        tool_choice = {"type": "function", "function": {"name": force_tool}}

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=0,
        )
    except Exception as e:
        print(f"Assistant: Sorry, I had trouble processing that. (Error: {e})")
        return

    response_message = response.choices[0].message

    if not response_message.tool_calls:
        stream_completion()
        return

    messages.append({
        "role": "assistant",
        "content": response_message.content,
        "tool_calls": [tc.model_dump() for tc in response_message.tool_calls],
    })

    for tool_call in response_message.tool_calls:
        if tool_call.function.name == "semantic_search":
            args = json.loads(tool_call.function.arguments)
            result = search_kb(knowledge_base, args.get("query", ""))
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        elif tool_call.function.name == "summarize_session":
            result = summarize_session(client, MODEL_NAME, messages)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

            print(f"Assistant: {result}")
            messages.append({"role": "assistant", "content": result})
            return

        else:
            result = "Unknown tool."
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    stream_completion()


def chat(user_input: str):
    """Append the user's message to the conversation and generate a reply."""
    messages.append({"role": "user", "content": user_input})
    get_completion()


def call_tool_forced(tool_name: str, query: str = ""):
    """Force the model to invoke a specific tool."""
    if query:
        messages.append({"role": "user", "content": f"Search for: {query}"})
    get_completion(force_tool=tool_name)


def stream_completion() -> str:
    """Stream the assistant's response token by token and return the complete text."""
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
    )

    print("Assistant: ", end="", flush=True)
    full_reply = ""

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            full_reply += delta

    print()
    messages.append({"role": "assistant", "content": full_reply})
    return full_reply


def save_session():
    """Save the current conversation and knowledge base to a JSON file."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"session_{timestamp}.json"

    file_path = SESSION_DIR / filename

    data = {
        "messages": messages,
        "knowledge_base": knowledge_base.to_dict(),
    }

    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Assistant: Session saved to {file_path.name}")


def load_session():
    filename = input("Enter filename to load session (e.g. session_2026-07-17_15-23-01.json): ").strip()

    if not filename:
        print("Assistant: Please provide a filename.")
        return

    file_path = Path(filename)
    if not file_path.is_absolute() and not file_path.exists():
        file_path = SESSION_DIR / filename

    if not file_path.exists():
        print(f"Assistant: File not found: {file_path}")
        return

    data = json.loads(file_path.read_text(encoding="utf-8"))

    messages.clear()
    messages.extend(data.get("messages", []))

    knowledge_base.load_from_dict(data.get("knowledge_base", {}))

    print(f"Assistant: Session loaded from {file_path.name}")
    print(f"[Info] {len(messages)} messages, {len(knowledge_base)} facts in KB")


def main():
    """Start the CLI assistant and process user commands."""
    global MODEL_NAME
    args = parse_args()
    MODEL_NAME = args.model

    print(f"Assistant is ready! Using model: {MODEL_NAME}")
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
            change_prompt(new_prompt_input)

        elif user_input == "/update_kb_text":
            update_kb_text()

        elif user_input == "/update_kb_voice":
            update_kb_voice()

        elif user_input == "/search":
            query = input("Enter your search query: ").strip()
            call_tool_forced("semantic_search", query=query)

        elif user_input == "/summarize_session":
            call_tool_forced("summarize_session")

        elif user_input == "/save_session":
            save_session()

        elif user_input == "/load_session":
            load_session()

        else:
            chat(user_input)


if __name__ == "__main__":
    main()

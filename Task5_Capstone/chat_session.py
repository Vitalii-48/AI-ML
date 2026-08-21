import json
from datetime import datetime
from pathlib import Path

from groq import Groq
from groq.types.chat import ChatCompletionMessage

from settings import settings
from schemas import tools
from constants import DEFAULT_MODEL_NAME, WHISPER_MODEL_NAME, AUDIO_DIR, SESSION_DIR
from prompts import SYSTEM_PROMPT
from vector_store import VectorStore
from tools import search_kb, summarize_session
from tts import speak


class ChatSession:
    """Owns the conversation history, the knowledge base, and talks to Groq."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, voice_enabled: bool = False):

        self.client = Groq(api_key=settings.groq_api_key)
        self.knowledge_base = VectorStore()
        self.model_name = model_name
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.voice_enabled = voice_enabled

    def update_kb_text(self, fact: str) -> None:
        """Add one fact typed by the user to the knowledge base."""
        self.knowledge_base.add(fact)

    def update_kb_voice(self, file_path: Path) -> str:
        """Transcribe an audio file with Whisper and store the result in the knowledge base."""
        if not file_path.is_absolute() and not file_path.exists():
            file_path = AUDIO_DIR / file_path

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                file=audio_file,
                model=WHISPER_MODEL_NAME,
            )

        text = transcription.text.strip()
        if not text:
            raise ValueError("Transcription was empty, nothing to add.")

        self.knowledge_base.add(text, metadata={"source": file_path.name})
        return text

    def change_prompt(self, new_prompt_input: str) -> str:
        """Update the system prompt from raw text or from a file."""
        file_path = Path(new_prompt_input)

        if file_path.exists() and file_path.is_file():
            new_prompt = file_path.read_text(encoding="utf-8").strip()
        else:
            new_prompt = new_prompt_input

        self.messages[0] = {"role": "system", "content": new_prompt}
        return new_prompt

    def _execute_tool_calls(self, response_message: ChatCompletionMessage) -> None:
        """Append the assistant's tool-call message and execute each tool."""
        self.messages.append({
            "role": "assistant",
            "content": response_message.content,
            "tool_calls": [tc.model_dump() for tc in response_message.tool_calls],
        })

        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "semantic_search":
                args = json.loads(tool_call.function.arguments)
                result = search_kb(self.knowledge_base, args.get("query", ""))

            elif tool_call.function.name == "summarize_session":
                result = summarize_session(self.client, self.model_name, self.messages)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
                print(f"Assistant: {result}")
                self.messages.append({"role": "assistant", "content": result})

                if self.voice_enabled:
                    speak(result)
                return

            else:
                result = "Unknown tool."

            self.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        self._stream_reply()

    def _stream_reply(self) -> str:
        """Stream the assistant's response token by token and return the complete text."""
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
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

        self.messages.append({"role": "assistant", "content": full_reply})

        if self.voice_enabled:
            speak(full_reply)

        return full_reply

    def _get_completion(self, tool_to_be_called: str | None = None) -> None:
        """Call the model, deciding automatically or forcing a specific tool."""
        tool_choice = "auto"
        if tool_to_be_called:
            tool_choice = {"type": "function", "function": {"name": tool_to_be_called}}

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=0,
        )

        response_message = response.choices[0].message

        if not response_message.tool_calls:
            self._stream_reply()
            return

        self._execute_tool_calls(response_message)

    def chat(self, user_input: str) -> None:
        """Append the user's message and generate a reply."""
        self.messages.append({"role": "user", "content": user_input})
        try:
            self._get_completion()
        except Exception as e:
            self.messages.pop()
            print(f"Assistant: Sorry, I had trouble processing that. (Error: {e})")

    def call_tool_forced(self, tool_name: str, query: str) -> None:
        """Force the model to invoke a specific tool."""
        if query:
            self.messages.append({"role": "user", "content": f"Search for: {query}"})
        try:
            self._get_completion(tool_to_be_called=tool_name)
        except Exception as e:
            if query:
                self.messages.pop()
            print(f"Assistant: Sorry, I had trouble processing that. (Error: {e})")

    def save_session(self) -> str:
        """Save the current conversation and knowledge base to a JSON file."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"session_{timestamp}.json"
        file_path = SESSION_DIR / filename

        data = {
            "messages": self.messages,
            "knowledge_base": self.knowledge_base.to_dict(),
        }
        file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return filename

    def load_session(self, filename: str) -> None:
        """Load a previously saved session file."""
        file_path = Path(filename)
        if not file_path.is_absolute() and not file_path.exists():
            file_path = SESSION_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        data = json.loads(file_path.read_text(encoding="utf-8"))

        self.messages.clear()
        self.messages.extend(data.get("messages", []))
        self.knowledge_base.load_from_dict(data.get("knowledge_base", {}))

    def handle_command(self, user_input: str) -> None:
        """Processes CLI slash commands or delegates regular text to chat."""
        if user_input == "/change_prompt":
            new_prompt_input = input("Enter new prompt or a filename: ").strip()
            if new_prompt_input:
                new_prompt = self.change_prompt(new_prompt_input)
                print(f'Assistant: System prompt updated.\nNew prompt: "{new_prompt}"')
            else:
                print("Assistant: Please provide a new prompt or a filename.")

        elif user_input == "/update_kb_text":
            fact = input("Enter your fact: ")
            self.update_kb_text(fact)
            print(f"[Saved] Fact saved. Total facts in KB: {len(self.knowledge_base.texts)}")

        elif user_input == "/update_kb_voice":
            file_input = input("Enter audio file path (e.g. test_audio1.mp3): ").strip()
            try:
                text = self.update_kb_voice(Path(file_input))
                print(f"[Transcribed] \"{text}\"")
                print(f"[Saved] Total facts in KB: {len(self.knowledge_base.texts)}")
            except (FileNotFoundError, ValueError) as e:
                print(f"Assistant: {e}")

        elif user_input == "/search":
            query = input("Enter your search query: ").strip()
            self.call_tool_forced("semantic_search", query=query)

        elif user_input == "/summarize_session":
            self.call_tool_forced("summarize_session", query="")

        elif user_input == "/save_session":
            filename = self.save_session()
            print(f"Assistant: Session saved to {filename}")

        elif user_input == "/load_session":
            filename = input("Enter filename to load session: ").strip()
            try:
                self.load_session(filename)
                print(f"Assistant: Session loaded from {filename}")
                print(f"[Info] {len(self.messages)} messages, {len(self.knowledge_base.texts)} facts in KB")
            except FileNotFoundError as e:
                print(f"Assistant: {e}")

        else:
            self.chat(user_input)

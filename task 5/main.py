# task 5\main.py
import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

knowledge_base = []

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

messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

client = Groq(api_key=os.environ["GROQ_API_KEY"])

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


def update_kb_text():
    fact = input("Enter your fact: ")
    knowledge_base.append(fact)
    print(f"[Saved] Факт збережено. Всього фактів у KB: {len(knowledge_base)}")


def search_kb(query: str) -> str:
    if not knowledge_base:
        return "Knowledge base is empty. Add facts first with /update_kb_text"

    query_words = set(query.lower().split())

    best_match = None
    best_score = 0

    for fact in knowledge_base:
        fact_words = set(fact.lower().split())
        score = len(fact_words & query_words)

        if score > best_score:
            best_score = score
            best_match = fact

    if best_match:
        return best_match
    else:
        return "No relevant fact found."


def summarize_session() -> str:
    """Generate a summary of the current chat session."""
    conversation = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role in ("user", "assistant") and content:
            conversation.append(f"{role.capitalize()}: {content}")

    if not conversation:
        return "No conversation to summarize."

    context = "\n".join(conversation)

    summary_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Summarize the following conversation.\n"
                    "Return a short bullet-point summary.\n"
                    "Include only what was discussed.\n"
                    "Do not invent facts.\n"
                    "Do not mention any knowledge base or external information."
                ),
            },
            {
                "role": "user",
                "content": context,
            },
        ],
        temperature=0.2,
    )

    return summary_response.choices[0].message.content


def get_completion(force_tool: str = None):
    """
    Робить виклик до GPT з tools.
    force_tool -- якщо вказано, примусово викликає саме цей tool (tool_choice).
                  Якщо None -- дає GPT самому вирішувати (tool_choice="auto").
    """
    tool_choice = "auto"
    if force_tool:
        tool_choice = {"type": "function", "function": {"name": force_tool}}

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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
        reply = response_message.content
        print(f"Assistant: {reply}")
        messages.append({"role": "assistant", "content": reply})
        return

    messages.append({
        "role": "assistant",
        "content": response_message.content,
        "tool_calls": [tc.model_dump() for tc in response_message.tool_calls],
    })

    for tool_call in response_message.tool_calls:
        if tool_call.function.name == "semantic_search":
            args = json.loads(tool_call.function.arguments)
            result = search_kb(args.get("query", ""))

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        elif tool_call.function.name == "summarize_session":
            result = summarize_session()

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

    final_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )
    final_reply = final_response.choices[0].message.content
    print(f"Assistant: {final_reply}")
    messages.append({"role": "assistant", "content": final_reply})


def chat(user_input: str):
    messages.append({"role": "user", "content": user_input})
    get_completion()


def call_tool_forced(tool_name: str, query: str = ""):
    if query:
        messages.append({"role": "user", "content": f"Search for: {query}"})
    get_completion(force_tool=tool_name)


def main():
    print("Assistant is ready! Commands: /update_kb_text, /search, /summarize_session, /q")

    while True:
        user_input = input("\n> You: ").strip()

        if user_input == "/q":
            print("Exiting...")
            break

        if user_input == "/update_kb_text":
            update_kb_text()

        elif user_input == "/search":
            query = input("Enter your search query: ").strip()
            call_tool_forced("semantic_search", query=query)

        elif user_input == "/summarize_session":
            call_tool_forced("summarize_session")

        else:
            chat(user_input)


if __name__ == "__main__":
    main()
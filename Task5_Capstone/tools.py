from groq import Groq

from vector_store import VectorStore
from prompts import SUMMARY_PROMPT


def search_kb(knowledge_base: VectorStore, query: str) -> str:
    """Search the knowledge base and return the most relevant fact."""
    if knowledge_base.is_empty():
        return "Knowledge base is empty. Add facts first with /update_kb_text"

    result = knowledge_base.search(query, top_k=1)

    if not result:
        return "No relevant facts found."

    best_fact, score, metadata = result[0]

    return best_fact


def summarize_session(client: Groq, model_name: str, messages: list) -> str:
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
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": SUMMARY_PROMPT,
            },
            {
                "role": "user",
                "content": context,
            },
        ],
        temperature=0.0,
    )
    return summary_response.choices[0].message.content

SEMANTIC_SEARCH_TOOL = {
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
}

SUMMARIZE_SESSION_TOOL = {
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

tools = [SEMANTIC_SEARCH_TOOL, SUMMARIZE_SESSION_TOOL]

# task_2\tool_schemas.py
from enums import ToolName

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": ToolName.CALCULATE.value,
            "description": "Solve math expressions like '5 * (2 + 3)'. Use only numbers and operators +,-,*,/,().",
            "parameters": {
                "type": "object",
                "properties": {
                    "expr": {"type": "string", "description": "The math expression to solve."}
                },
                "required": ["expr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": ToolName.EXPLAIN.value,
            "description": "Explain a study topic in simple terms (e.g., 'photosynthesis', 'gravity').",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The educational topic to explain."}
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": ToolName.FAKE_LOOKUP.value,
            "description": "Look up information in the local reference database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The term or topic to look up, e.g. 'python' or 'einstein'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": ToolName.WIKIPEDIA_SEARCH.value,
            "description": "Search the live Wikipedia API. Use ONLY when the user explicitly asks to search Wikipedia.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search term to look up on Wikipedia."}
                },
                "required": ["query"],
            },
        },
    },
]
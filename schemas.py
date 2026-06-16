analyze_user_state_schema = {
    "type": "function",
    "function": {
        "name": "analyze_user_state",
        "description": "Analyze the user's emotional state, viewing need, genre preferences, and avoidance preferences from natural language input.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_input": {
                    "type": "string",
                    "description": "The user's natural language request"
                }
            },
            "required": ["user_input"],
            "additionalProperties": False
        }
    }
}

retrieve_movies_schema = {
    "type": "function",
    "function": {
        "name": "retrieve_movies",
        "description": "Retrieve candidate movies based on user emotional state and preferences.",
        "parameters": {
            "type": "object",
            "properties": {
                "state": {
                    "type": "object",
                    "description": "Structured user state including emotion, need, tone, preferred genres, and avoid preferences"
                }
            },
            "required": ["state"],
            "additionalProperties": False
        }
    }
}

rag_retrieve_movies_schema = {
    "type": "function",
    "function": {
        "name": "rag_retrieve_movies",
        "description": "Retrieve movie candidates with lightweight RAG. It searches movie metadata and descriptions, applies user constraints, and returns evidence snippets for grounded recommendations.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_input": {
                    "type": "string",
                    "description": "The user's original natural language request"
                },
                "state": {
                    "type": "object",
                    "description": "Structured user state including emotion, need, tone, preferred genres, and avoid preferences"
                },
                "top_k": {
                    "type": "integer",
                    "description": "The number of candidate movies to retrieve",
                    "default": 5
                }
            },
            "required": ["user_input", "state"],
            "additionalProperties": False
        }
    }
}

TOOLS = [
    analyze_user_state_schema,
    retrieve_movies_schema,
    rag_retrieve_movies_schema
]

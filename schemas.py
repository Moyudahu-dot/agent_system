# Tool schema 是给大模型看的“工具说明书”。
# 模型会根据 name、description、parameters 判断什么时候调用哪个工具，
# 并按照 parameters 中定义的 JSON 格式生成工具参数。


# 用户状态分析工具：
# 输入用户原始文本，输出情绪、需求、偏好类型和避雷项。
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
            # 不允许模型额外传入未定义字段，可以减少参数漂移。
            "additionalProperties": False
        }
    }
}

# 早期规则检索工具：
# 现在主要用于对比，主流程使用 rag_retrieve_movies。
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

# RAG 检索工具：
# 模型调用这个工具时，需要同时传入用户原始输入和结构化 state。
# user_input 保留自然语言细节，state 提供情绪/偏好等结构化信号。
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

# 传给 DeepSeek/OpenAI SDK 的工具列表。
# main.py 会把这个列表放到 chat.completions.create(..., tools=TOOLS) 中。
TOOLS = [
    analyze_user_state_schema,
    retrieve_movies_schema,
    rag_retrieve_movies_schema
]

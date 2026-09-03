import json
import os
from openai import OpenAI

from schemas import TOOLS
from dispatcher import dispatch_tool
from agent.memory import SimpleMemory


# 全局 Memory：当前版本用内存字典保存用户偏好。
# 注意：程序退出后 Memory 会丢失，后续可以换成数据库或本地文件做持久化。
memory = SimpleMemory()

def create_deepseek_client() -> OpenAI:
    """
    Create a DeepSeek client through the OpenAI-compatible SDK.

    The OpenAI SDK requires an API key when creating the client. For this
    project, DEEPSEEK_API_KEY is clearer than OPENAI_API_KEY, so we support both:
    - DEEPSEEK_API_KEY: recommended for this project
    - OPENAI_API_KEY: fallback for OpenAI-compatible SDK behavior
    """
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set DEEPSEEK_API_KEY before running this project. "
            "PowerShell example: $env:DEEPSEEK_API_KEY='your_api_key'"
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )


client = create_deepseek_client()


def run_agent(user_input: str):
    """
    Agent 主入口。

    整体流程：
    1. 将用户输入和历史 Memory 组织成 messages。
    2. 把工具 schema 传给大模型，让模型自行决定调用哪个工具。
    3. 执行模型返回的 tool_calls，并把工具结果写回 messages。
    4. 模型基于工具结果继续推理，最终生成推荐回答。
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a movie recommendation agent. "
                f"User memory: {memory.get_memory()}. "
                # 这里强约束工具调用顺序：
                # 先理解用户状态，再做 RAG 检索，最后基于 evidence 生成推荐。
                "You must first call analyze_user_state, "
                "then call rag_retrieve_movies, "
                "then generate a recommendation grounded in the returned evidence."
            )
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    # 最多循环 5 轮，避免模型反复调用工具导致死循环。
    # 一般情况下：状态分析 -> RAG 检索 -> 最终回答，2-3 轮即可结束。
    for _ in range(5):
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        message = response.choices[0].message

        # 如果模型没有继续请求工具调用，说明它已经生成最终回答。
        if not message.tool_calls:
            return message.content

        # OpenAI/DeepSeek 的 Tool Calling 协议要求：
        # assistant 的 tool_calls 消息必须先加入 messages，
        # 后面才能追加对应的 tool 结果。
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
        )

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"\n[Tool Call] {tool_name} -> {tool_args}")

            tool_result = dispatch_tool(tool_name, tool_args)

            # 用户状态分析成功后，写入 Memory。
            # Memory 会在下一轮对话中进入 system prompt，帮助 Agent 保持个性化。
            if tool_name == "analyze_user_state" and tool_result["status"] == "ok":
                memory.update_from_state(tool_result)

            # 检索成功后，记录最近推荐候选，避免后续多轮对话重复推荐。
            if tool_name in ["retrieve_movies", "rag_retrieve_movies"] and tool_result["status"] == "ok":
                memory.add_recommendations(tool_result["candidates"])

            print(f"[Tool Result] {tool_result}")

            # 将工具结果作为 tool message 传回模型。
            # 模型下一步会读取这里的 candidates/evidence，并生成 grounded recommendation。
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, ensure_ascii=False)
                }
            )

    return "Agent stopped after max steps."


if __name__ == "__main__":
    # 命令行交互入口。
    user_input = input("User: ")
    answer = run_agent(user_input)
    print("\nAssistant:", answer)

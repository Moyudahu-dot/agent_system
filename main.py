import json
from openai import OpenAI

from schemas import TOOLS
from dispatcher import dispatch_tool
from agent.memory import SimpleMemory

memory = SimpleMemory()

client = OpenAI(
    base_url="https://api.deepseek.com"
)


def run_agent(user_input: str):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a movie recommendation agent. "
                f"User memory: {memory.get_memory()}. "
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

    for _ in range(5):
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        message = response.choices[0].message

        if not message.tool_calls:
            return message.content

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

            if tool_name == "analyze_user_state" and tool_result["status"] == "ok":
                memory.update_from_state(tool_result)

            if tool_name in ["retrieve_movies", "rag_retrieve_movies"] and tool_result["status"] == "ok":
                memory.add_recommendations(tool_result["candidates"])

            print(f"[Tool Result] {tool_result}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, ensure_ascii=False)
                }
            )

    return "Agent stopped after max steps."


if __name__ == "__main__":
    user_input = input("User: ")
    answer = run_agent(user_input)
    print("\nAssistant:", answer)

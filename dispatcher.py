from tools.user_state_tool import analyze_user_state
from tools.retrieve_tool import retrieve_movies
from tools.rag_retrieve_tool import rag_retrieve_movies


def dispatch_tool(tool_name: str, args: dict) -> dict:
    """
    Tool 分发器。

    大模型只会返回工具名和 JSON 参数，真正的 Python 函数调用由这里完成。
    这样可以把“模型决策”和“本地业务逻辑执行”解耦：
    - schemas.py 负责告诉模型有哪些工具可以用；
    - dispatcher.py 负责把工具名映射到具体函数；
    - tools/ 目录负责实现每个工具的业务逻辑。
    """
    try:
        print(f"[Dispatcher] tool={tool_name}, args={args}")

        # 分析用户情绪、观看需求、偏好类型和避雷项。
        if tool_name == "analyze_user_state":
            return analyze_user_state(**args)

        # 早期规则检索工具，保留用于对比，不是当前主流程的重点。
        if tool_name == "retrieve_movies":
            return retrieve_movies(**args)

        # 当前主检索工具：BGE embedding + FAISS 向量召回 + 规则过滤。
        if tool_name == "rag_retrieve_movies":
            return rag_retrieve_movies(**args)

        # 如果模型调用了未注册工具，返回结构化错误，避免程序崩溃。
        return {
            "status": "error",
            "message": f"Unknown tool: {tool_name}"
        }

    except TypeError as e:
        # 参数名缺失、参数类型不对等问题通常会走到这里。
        return {
            "status": "error",
            "message": f"Invalid arguments for tool '{tool_name}': {str(e)}"
        }

    except Exception as e:
        # 兜底异常处理：保证工具失败时仍然以 JSON 形式返回给模型。
        return {
            "status": "error",
            "message": f"Tool execution failed: {str(e)}"
        }

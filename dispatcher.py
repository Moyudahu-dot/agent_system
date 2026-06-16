from tools.user_state_tool import analyze_user_state
from tools.retrieve_tool import retrieve_movies
from tools.rag_retrieve_tool import rag_retrieve_movies


def dispatch_tool(tool_name: str, args: dict) -> dict:
    """
    Dispatch the tool call to the correct Python function.
    """
    try:
        print(f"[Dispatcher] tool={tool_name}, args={args}")

        if tool_name == "analyze_user_state":
            return analyze_user_state(**args)

        if tool_name == "retrieve_movies":
            return retrieve_movies(**args)

        if tool_name == "rag_retrieve_movies":
            return rag_retrieve_movies(**args)

        return {
            "status": "error",
            "message": f"Unknown tool: {tool_name}"
        }

    except TypeError as e:
        return {
            "status": "error",
            "message": f"Invalid arguments for tool '{tool_name}': {str(e)}"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Tool execution failed: {str(e)}"
        }

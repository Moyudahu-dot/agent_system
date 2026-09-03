import pandas as pd


def retrieve_movies(state: dict) -> dict:
    """
    早期规则版检索工具。

    当前主流程已经切换到 tools/rag_retrieve_tool.py 中的 BGE + FAISS RAG。
    这个函数保留用于对比：它不做语义向量检索，只根据 CSV 字段和规则过滤电影。
    """
    try:
        # 读取根目录下的小型电影数据集。
        df = pd.read_csv("movies.csv")

        candidates = []

        for _, row in df.iterrows():
            match = True

            # 1. 避雷项过滤：用户不想看沉重内容时，过滤 high heaviness。
            if "heavy" in state.get("avoid", []):
                if row["heaviness"] == "high":
                    match = False

            # 2. 情绪匹配：根据用户希望的 tone 映射到电影 mood。
            if state.get("tone") == "healing":
                if row["mood"] not in ["warm", "light", "magical"]:
                    match = False

            if state.get("tone") == "fun":
                if row["mood"] not in ["light", "warm"]:
                    match = False

            # 3. 类型匹配：如果用户明确喜欢某类影片，候选电影类型中需要包含它。
            preferred = state.get("preferred_genres", [])
            if preferred:
                if not any(g in row["genre"] for g in preferred):
                    match = False

            if match:
                candidates.append({
                    "title": row["title"],
                    "genre": row["genre"],
                    "mood": row["mood"],
                    "description": row["description"]
                })

        return {
            "status": "ok",
            "candidates": candidates[:5]  # 最多返回 5 个候选。
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

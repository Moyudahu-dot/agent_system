import pandas as pd


def retrieve_movies(state: dict) -> dict:
    """
    Retrieve candidate movies based on user emotional state and preferences.
    """
    try:
        df = pd.read_csv("movies.csv")

        candidates = []

        for _, row in df.iterrows():
            match = True

            # 1️⃣ 避免项过滤（最重要）
            if "heavy" in state.get("avoid", []):
                if row["heaviness"] == "high":
                    match = False

            # 2️⃣ 情绪匹配（简单版本）
            if state.get("tone") == "healing":
                if row["mood"] not in ["warm", "light", "magical"]:
                    match = False

            if state.get("tone") == "fun":
                if row["mood"] not in ["light", "warm"]:
                    match = False

            # 3️⃣ 类型偏好匹配
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
            "candidates": candidates[:5]  # 最多返回5个
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
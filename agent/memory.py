class SimpleMemory:
    def __init__(self):
        # 当前版本使用最简单的短期记忆结构。
        # 这些信息会被拼进 system prompt，让 Agent 在多轮对话中知道用户偏好。
        self.data = {
            "liked_genres": [],
            "disliked_genres": [],
            "recent_recommendations": [],
            "last_emotion": ""
        }

    def update_from_state(self, state: dict):
        # 更新最近一次识别到的用户情绪，例如 stressed、sad、happy。
        self.data["last_emotion"] = state.get("emotion", "")

        # 将本轮识别出的偏好类型加入长期偏好列表，避免重复添加。
        for g in state.get("preferred_genres", []):
            if g not in self.data["liked_genres"]:
                self.data["liked_genres"].append(g)

        # 将用户明确避雷的内容记录下来，例如 heavy、slow、romance。
        for g in state.get("avoid", []):
            if g not in self.data["disliked_genres"]:
                self.data["disliked_genres"].append(g)

    def add_recommendations(self, movies: list):
        # 只记录最近一批推荐标题，用于后续扩展“避免重复推荐”。
        titles = [m["title"] for m in movies]
        self.data["recent_recommendations"] = titles[:5]

    def get_memory(self):
        # 返回完整 Memory，供 main.py 写入 system prompt。
        return self.data

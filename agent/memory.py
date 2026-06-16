class SimpleMemory:
    def __init__(self):
        self.data = {
            "liked_genres": [],
            "disliked_genres": [],
            "recent_recommendations": [],
            "last_emotion": ""
        }

    def update_from_state(self, state: dict):
        # 更新情绪
        self.data["last_emotion"] = state.get("emotion", "")

        # 更新偏好
        for g in state.get("preferred_genres", []):
            if g not in self.data["liked_genres"]:
                self.data["liked_genres"].append(g)

        for g in state.get("avoid", []):
            if g not in self.data["disliked_genres"]:
                self.data["disliked_genres"].append(g)

    def add_recommendations(self, movies: list):
        titles = [m["title"] for m in movies]
        self.data["recent_recommendations"] = titles[:5]

    def get_memory(self):
        return self.data
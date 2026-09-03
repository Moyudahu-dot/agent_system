def analyze_user_state(user_input: str) -> dict:
    """
    从自然语言输入中抽取用户状态。

    这个工具相当于 Agent 的“理解层”，负责把用户一句话转成结构化信息：
    - emotion: 用户情绪
    - need: 观看需求
    - tone: 希望影片带来的感受
    - preferred_genres: 偏好的电影类型
    - avoid: 明确不想看的内容

    当前版本用关键词规则实现，优点是可解释、稳定、方便调试。
    后续可以替换成 LLM 结构化抽取或训练一个轻量分类器。
    """
    text = user_input.lower()

    # 默认状态：如果没有识别到明显信号，就按普通推荐处理。
    emotion = "neutral"
    need = "general"
    tone = "neutral"
    preferred_genres = []
    avoid = []

    # 情绪/需求识别：根据关键词把用户输入映射成结构化状态。
    if "stress" in text or "tired" in text or "pressure" in text:
        emotion = "stressed"
        need = "relaxing"
        tone = "healing"

    if "sad" in text or "upset" in text or "heartbroken" in text:
        emotion = "sad"
        need = "comfort"
        tone = "warm"

    if "happy" in text or "excited" in text or "great" in text:
        emotion = "happy"
        need = "entertaining"
        tone = "fun"

    # 类型偏好识别：这些类型会参与 RAG query 构造，提高召回相关性。
    if "comedy" in text:
        preferred_genres.append("comedy")
    if "sci-fi" in text or "science fiction" in text:
        preferred_genres.append("sci-fi")
    if "romance" in text:
        preferred_genres.append("romance")
    if "drama" in text:
        preferred_genres.append("drama")
    if "animation" in text:
        preferred_genres.append("animation")
    if "thriller" in text:
        preferred_genres.append("thriller")
    if "fantasy" in text:
        preferred_genres.append("fantasy")

    # 避雷项识别：用户明确不要的内容属于硬约束。
    # 后续在 RAG 召回后会用 passes_constraints 做过滤。
    if "don't want" in text or "do not want" in text or "avoid" in text or "not too" in text:
        if "heavy" in text or "depressing" in text or "dark" in text:
            avoid.append("heavy")
        if "violent" in text:
            avoid.append("violent")
        if "slow" in text:
            avoid.append("slow")
        if "romance" in text:
            avoid.append("romance")

    # 工具统一返回 dict，方便 main.py 直接序列化给 LLM。
    return {
        "status": "ok",
        "emotion": emotion,
        "need": need,
        "tone": tone,
        "preferred_genres": preferred_genres,
        "avoid": avoid
    }


if __name__ == "__main__":
    # 本文件可单独运行，方便测试状态分析工具是否符合预期。
    result = analyze_user_state("I feel stressed and want a light comedy, but nothing too dark.")
    print(result)

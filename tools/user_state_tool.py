def analyze_user_state(user_input: str) -> dict:
    """
    Analyze the user's emotional state and recommendation preferences
    from natural language input.
    """
    text = user_input.lower()

    emotion = "neutral"
    need = "general"
    tone = "neutral"
    preferred_genres = []
    avoid = []

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

    if "don't want" in text or "do not want" in text or "avoid" in text or "not too" in text:
        if "heavy" in text or "depressing" in text or "dark" in text:
            avoid.append("heavy")
        if "violent" in text:
            avoid.append("violent")
        if "slow" in text:
            avoid.append("slow")
        if "romance" in text:
            avoid.append("romance")

    return {
        "status": "ok",
        "emotion": emotion,
        "need": need,
        "tone": tone,
        "preferred_genres": preferred_genres,
        "avoid": avoid
    }


if __name__ == "__main__":
    result = analyze_user_state("I feel stressed and want a light comedy, but nothing too dark.")
    print(result)
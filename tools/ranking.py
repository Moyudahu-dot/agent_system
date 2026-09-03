"""
Rerank module for movie candidates.

The first-stage retriever uses BGE + FAISS to recall semantically related
movies. Rerank is the second-stage scorer: it reorders those recalled
candidates with task-specific signals such as genre preference, mood fit,
pace fit, heaviness, and user constraints.
"""


HEALING_MOODS = {"warm", "light", "heartwarming", "magical", "inspiring"}
FUN_MOODS = {"light", "warm"}
COMFORT_MOODS = {"warm", "heartwarming", "inspiring", "magical"}


def has_genre_match(movie: dict, preferred_genres: list[str]) -> bool:
    genre_text = str(movie.get("genre", "")).lower()
    return any(genre.lower() in genre_text for genre in preferred_genres)


def mood_match_score(movie: dict, state: dict) -> float:
    mood = str(movie.get("mood", "")).lower()
    tone = str(state.get("tone", "")).lower()
    need = str(state.get("need", "")).lower()

    if tone == "healing" and mood in HEALING_MOODS:
        return 1.0
    if tone == "fun" and mood in FUN_MOODS:
        return 1.0
    if need == "comfort" and mood in COMFORT_MOODS:
        return 1.0
    if tone and tone == mood:
        return 1.0
    return 0.0


def pace_match_score(movie: dict, state: dict) -> float:
    pace = str(movie.get("pace", "")).lower()
    need = str(state.get("need", "")).lower()
    tone = str(state.get("tone", "")).lower()

    if need == "relaxing" or tone == "healing":
        return 1.0 if pace in {"medium", "slow"} else 0.0
    if need == "entertaining" or tone == "fun":
        return 1.0 if pace in {"medium", "fast"} else 0.0
    return 0.5


def constraint_penalty(movie: dict, state: dict) -> float:
    """
    Soft penalty for risky candidates.

    Hard filtering is still done in rag_retrieve_tool.py. This penalty is kept
    as an extra safety net and also makes rerank scores more interpretable.
    """
    avoid = state.get("avoid", [])
    penalty = 0.0

    if "heavy" in avoid and movie.get("heaviness") == "high":
        penalty += 1.0
    if "slow" in avoid and movie.get("pace") == "slow":
        penalty += 0.8
    if "romance" in avoid and "romance" in str(movie.get("genre", "")).lower():
        penalty += 0.8

    return penalty


def rerank_movies(candidates: list[dict], state: dict, top_k: int = 5) -> list[dict]:
    """
    Rerank recalled candidates with a weighted scoring function.

    Score design:
    - vector_score: semantic relevance from BGE + FAISS
    - genre_score: whether the movie genre matches explicit user preference
    - mood_score: whether movie mood fits user's desired tone
    - pace_score: whether movie pace fits user's current need
    - lightness_score: prefer low-heaviness content for relaxing/healing needs
    - penalty: punish candidates that conflict with user constraints
    """
    preferred_genres = state.get("preferred_genres", [])
    reranked = []

    for movie in candidates:
        vector_score = float(movie.get("retrieval_score", 0.0))
        genre_score = 1.0 if has_genre_match(movie, preferred_genres) else 0.0
        mood_score = mood_match_score(movie, state)
        pace_score = pace_match_score(movie, state)
        lightness_score = 1.0 if movie.get("heaviness") == "low" else 0.0
        penalty = constraint_penalty(movie, state)

        final_score = (
            0.55 * vector_score
            + 0.20 * genre_score
            + 0.15 * mood_score
            + 0.07 * pace_score
            + 0.03 * lightness_score
            - penalty
        )

        reranked_movie = {
            **movie,
            "rerank_score": round(final_score, 4),
            "rerank_features": {
                "vector_score": round(vector_score, 4),
                "genre_match": genre_score,
                "mood_match": mood_score,
                "pace_match": pace_score,
                "lightness": lightness_score,
                "constraint_penalty": penalty,
            },
        }
        reranked.append(reranked_movie)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return reranked[:top_k]

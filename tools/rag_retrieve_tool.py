import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE_DIR = os.path.join(BASE_DIR, "rag_store")
INDEX_PATH = os.path.join(STORE_DIR, "movies_bge.faiss")
METADATA_PATH = os.path.join(STORE_DIR, "movies_metadata.pkl")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_model = None
_index = None
_metadata = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def load_rag_store() -> tuple[faiss.Index, list[dict]]:
    global _index, _metadata
    if _index is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(
                f"Missing FAISS index: {INDEX_PATH}. Run agent/build_bge_index.py first."
            )
        _index = faiss.read_index(INDEX_PATH)

    if _metadata is None:
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(
                f"Missing RAG metadata: {METADATA_PATH}. Run agent/build_bge_index.py first."
            )
        with open(METADATA_PATH, "rb") as f:
            _metadata = pickle.load(f)

    return _index, _metadata


def build_query(user_input: str, state: dict) -> str:
    state_terms = []
    for key in ["emotion", "need", "tone"]:
        value = state.get(key)
        if value:
            state_terms.append(str(value))
    state_terms.extend(state.get("preferred_genres", []))
    return " ".join([user_input, *state_terms])


def passes_constraints(movie: dict, state: dict) -> bool:
    avoid = state.get("avoid", [])
    if "heavy" in avoid and movie["heaviness"] == "high":
        return False
    if "slow" in avoid and movie["pace"] == "slow":
        return False
    if "romance" in avoid and "romance" in movie["genre"]:
        return False
    return True


def to_candidate(movie: dict, score: float) -> dict:
    return {
        "title": movie["title"],
        "genre": movie["genre"],
        "mood": movie["mood"],
        "pace": movie["pace"],
        "heaviness": movie["heaviness"],
        "description": movie["description"],
        "retrieval_score": round(float(score), 4),
        "evidence": movie["document"],
    }


def rag_retrieve_movies(user_input: str, state: dict, top_k: int = 5) -> dict:
    """
    Retrieve movie candidates from a BGE embedding + FAISS vector index.
    """
    try:
        index, metadata = load_rag_store()
        model = get_model()

        query = build_query(user_input, state)
        query_vector = model.encode([query], normalize_embeddings=True)
        query_vector = np.array(query_vector).astype("float32")

        search_k = min(max(top_k * 3, top_k), len(metadata))
        scores, indices = index.search(query_vector, search_k)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            movie = metadata[int(idx)]
            if not passes_constraints(movie, state):
                continue
            candidates.append(to_candidate(movie, score))
            if len(candidates) >= top_k:
                break

        return {
            "status": "ok",
            "query": query,
            "retrieval_method": "bge-small-en-v1.5 + faiss",
            "candidates": candidates,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }

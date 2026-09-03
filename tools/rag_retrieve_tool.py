import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from tools.ranking import rerank_movies


# BASE_DIR 指向项目根目录，保证无论从哪里运行脚本，都能找到 rag_store。
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE_DIR = os.path.join(BASE_DIR, "rag_store")
INDEX_PATH = os.path.join(STORE_DIR, "movies_bge.faiss")
METADATA_PATH = os.path.join(STORE_DIR, "movies_metadata.pkl")

# BGE 是一个通用文本 embedding 模型，用来把 query 和电影内容映射到同一个向量空间。
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# 简单的进程内缓存，避免每次工具调用都重新加载模型和索引。
_model = None
_index = None
_metadata = None


def get_model() -> SentenceTransformer:
    """
    懒加载 embedding 模型。

    BGE 模型加载比较耗时，所以第一次调用时加载，后续复用全局变量。
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def load_rag_store() -> tuple[faiss.Index, list[dict]]:
    """
    加载离线构建好的 FAISS 索引和电影元数据。

    - movies_bge.faiss 保存电影文本向量索引；
    - movies_metadata.pkl 保存每个向量对应的电影标题、类型、简介等信息。
    """
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
    """
    构造检索 query。

    只使用用户原始输入可能会漏掉 analyze_user_state 提取出的结构化信息，
    所以这里把 emotion、need、tone、preferred_genres 一起拼进去，
    让向量召回更贴近用户真实需求。
    """
    state_terms = []
    for key in ["emotion", "need", "tone"]:
        value = state.get(key)
        if value:
            state_terms.append(str(value))
    state_terms.extend(state.get("preferred_genres", []))
    return " ".join([user_input, *state_terms])


def passes_constraints(movie: dict, state: dict) -> bool:
    """
    硬约束过滤。

    向量召回负责“语义相关”，但它不一定严格遵守用户避雷项。
    因此召回后需要再做规则过滤，避免推荐用户明确不想看的内容。
    """
    avoid = state.get("avoid", [])
    if "heavy" in avoid and movie["heaviness"] == "high":
        return False
    if "slow" in avoid and movie["pace"] == "slow":
        return False
    if "romance" in avoid and "romance" in movie["genre"]:
        return False
    return True


def to_candidate(movie: dict, score: float) -> dict:
    """
    将 metadata 中的一条电影记录转换成 Agent 可消费的候选结果。

    evidence 会传回 LLM，要求模型基于这些证据生成推荐理由，
    这样可以减少模型凭空编造信息的风险。
    """
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
    使用 BGE embedding + FAISS 向量索引召回电影候选。

    当前流程：
    1. 加载 FAISS 索引和电影 metadata；
    2. 将用户 query 编码成向量；
    3. 从 FAISS 中召回相似度最高的候选；
    4. 根据用户避雷项做硬过滤；
    5. 返回候选电影、相似度分数和 evidence。
    """
    try:
        index, metadata = load_rag_store()
        model = get_model()

        query = build_query(user_input, state)

        # normalize_embeddings=True 会把向量归一化。
        # 因为建索引时也做了归一化，所以 FAISS IndexFlatIP 的内积近似等价于 cosine similarity。
        query_vector = model.encode([query], normalize_embeddings=True)
        query_vector = np.array(query_vector).astype("float32")

        # 先召回 top_k 的多倍，再做规则过滤。
        # 这样可以避免前 top_k 中有部分电影被过滤后，最终候选数量不足。
        search_k = min(max(top_k * 3, top_k), len(metadata))
        scores, indices = index.search(query_vector, search_k)

        recalled_candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            movie = metadata[int(idx)]
            if not passes_constraints(movie, state):
                continue
            recalled_candidates.append(to_candidate(movie, score))

        candidates = rerank_movies(recalled_candidates, state, top_k)

        return {
            "status": "ok",
            "query": query,
            "retrieval_method": "bge-small-en-v1.5 + faiss",
            "rerank_method": "weighted vector + genre + mood + pace + constraint score",
            "candidates": candidates,
        }

    except Exception as e:
        # 工具失败时仍然返回结构化错误，方便 LLM 或上层逻辑处理。
        return {
            "status": "error",
            "message": str(e),
        }

import os
import pickle

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# BASE_DIR 指向项目根目录。
# build_bge_index.py 位于 agent/ 目录下，所以 dirname(dirname(__file__)) 才是项目根目录。
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 用 data/movies.csv 作为知识库数据源。
CSV_PATH = os.path.join(BASE_DIR, "data", "movies.csv")

# rag_store 用来保存离线生成的向量索引和 metadata。
STORE_DIR = os.path.join(BASE_DIR, "rag_store")
INDEX_PATH = os.path.join(STORE_DIR, "movies_bge.faiss")
METADATA_PATH = os.path.join(STORE_DIR, "movies_metadata.pkl")

# BGE-small 是轻量级英文 embedding 模型，适合当前英文电影简介数据。
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def build_document(row: pd.Series) -> str:
    """
    将一行电影数据拼接成可被 embedding 模型编码的文本。

    RAG 检索不是只看 description，而是把 title、genre、mood、pace、
    heaviness 等结构化字段都放进去，让向量表示包含更多推荐相关信息。
    """
    return (
        f"Title: {row['title']}\n"
        f"Genre: {row['genre']}\n"
        f"Mood: {row['mood']}\n"
        f"Pace: {row['pace']}\n"
        f"Heaviness: {row['heaviness']}\n"
        f"Description: {row['description']}"
    )


def main():
    """
    离线构建 RAG 向量索引。

    这个脚本通常在数据更新后运行一次即可。
    在线检索时 tools/rag_retrieve_tool.py 会直接读取生成好的索引文件。
    """
    os.makedirs(STORE_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    documents = []
    metadata = []

    for _, row in df.iterrows():
        doc = build_document(row)
        documents.append(doc)

        # metadata 用于在 FAISS 返回向量下标后，找回原始电影信息。
        metadata.append({
            "title": row["title"],
            "genre": row["genre"],
            "mood": row["mood"],
            "pace": row["pace"],
            "heaviness": row["heaviness"],
            "description": row["description"],
            "document": doc,
        })

    print("Loading BGE embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Generating embeddings...")
    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        # 归一化后，向量内积可以近似等价于 cosine similarity。
        normalize_embeddings=True
    )

    vectors = np.array(embeddings).astype("float32")

    dimension = vectors.shape[1]

    # IndexFlatIP 是精确内积检索。
    # 当前数据量小，精确检索足够；数据量大时可以替换成 IVF、HNSW 等近似索引。
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    # 保存 FAISS 索引。
    faiss.write_index(index, INDEX_PATH)

    # 保存向量下标对应的电影信息。
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("BGE FAISS index built successfully.")
    print(f"Total movies indexed: {len(metadata)}")
    print(f"Index saved to: {INDEX_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()

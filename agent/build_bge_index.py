import os
import pickle

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(BASE_DIR, "data", "movies.csv")
STORE_DIR = os.path.join(BASE_DIR, "rag_store")

INDEX_PATH = os.path.join(STORE_DIR, "movies_bge.faiss")
METADATA_PATH = os.path.join(STORE_DIR, "movies_metadata.pkl")#存原电影信息

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

#分片逻辑
def build_document(row: pd.Series) -> str:
    return (
        f"Title: {row['title']}\n"
        f"Genre: {row['genre']}\n"
        f"Mood: {row['mood']}\n"
        f"Pace: {row['pace']}\n"
        f"Heaviness: {row['heaviness']}\n"
        f"Description: {row['description']}"
    )


def main():
    os.makedirs(STORE_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    documents = []
    metadata = []

    for _, row in df.iterrows():
        doc = build_document(row)
        documents.append(doc)

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
        normalize_embeddings=True
    )

    vectors = np.array(embeddings).astype("float32")

    dimension = vectors.shape[1]

    # 向量已经 normalize，所以 IndexFlatIP 等价于 cosine similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("BGE FAISS index built successfully.")
    print(f"Total movies indexed: {len(metadata)}")
    print(f"Index saved to: {INDEX_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()
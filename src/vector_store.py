# src/vector_store.py

import os
import faiss
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(FAISS_DIR, "index.pkl")


def save_vector_store(embeddings, chunks):
    """Save embeddings into FAISS index"""
    print("Saving vector store...")

    os.makedirs(FAISS_DIR, exist_ok=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved {index.ntotal} vectors to faiss_index/")


def load_vector_store():
    """Load FAISS index from disk"""
    print("Loading vector store...")

    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            "Vector store not found. Please process a document first."
        )

    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    print(f"Loaded {index.ntotal} vectors")
    return index, chunks


def search_similar_chunks(query_embedding, index, chunks, top_k=3):
    """Find top 3 most similar chunks to the query"""
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in indices[0]:
        if i != -1:
            results.append(chunks[i])

    return results
# src/embeddings.py

from sentence_transformers import SentenceTransformer

# load the model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embeddings(chunks):
    """Convert list of text chunks into vectors"""
    print(f"Creating embeddings for {len(chunks)} chunks...")

    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def get_query_embedding(query):
    """Convert a single question into a vector"""
    embedding = model.encode([query], convert_to_numpy=True)
    return embedding
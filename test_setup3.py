# test_setup.py

from src.pdf_loader import load_and_chunk_pdf
from src.embeddings import get_embeddings, get_query_embedding
from src.vector_store import save_vector_store, load_vector_store, search_similar_chunks

# Step 1 - load and chunk PDF
chunks = load_and_chunk_pdf(r"C:\Users\janit\OneDrive\Desktop\DocuSearch-AI\data\uploaded_docs\test.pdf")

# Step 2 - create embeddings
embeddings = get_embeddings(chunks)

# Step 3 - save to FAISS
save_vector_store(embeddings, chunks)

# Step 4 - test a search query
index, chunks = load_vector_store()
query = "what is this document about?"
query_embedding = get_query_embedding(query)
results = search_similar_chunks(query_embedding, index, chunks)

print("\n--- Top matching chunks for your query ---")
for i, result in enumerate(results):
    print(f"\nResult {i+1}:")
    print(result)
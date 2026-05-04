# test_setup.py

from src.pdf_loader import load_and_chunk_pdf
from src.embeddings import get_embeddings, get_query_embedding
from src.vector_store import save_vector_store, load_vector_store, search_similar_chunks
from src.llm_handler import get_answer

# Step 1 - load and chunk PDF
chunks = load_and_chunk_pdf(r"C:\Users\janit\OneDrive\Desktop\DocuSearch-AI\data\uploaded_docs\test.pdf")

# Step 2 - create embeddings
embeddings = get_embeddings(chunks)

# Step 3 - save to FAISS
save_vector_store(embeddings, chunks)

# Step 4 - load vector store
index, chunks = load_vector_store()

# Step 5 - ask a question
query = "what is this document about?"
query_embedding = get_query_embedding(query)

# Step 6 - find relevant chunks
relevant_chunks = search_similar_chunks(query_embedding, index, chunks)

# Step 7 - get answer from Phi
answer = get_answer(query, relevant_chunks)

print("\n--- Question ---")
print(query)
print("\n--- Answer ---")
print(answer)
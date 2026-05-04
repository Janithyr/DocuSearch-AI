# test_setup.py

from src.pdf_loader import load_and_chunk_pdf

# your exact path
file_path = r"C:\Users\janit\OneDrive\Desktop\DocuSearch-AI\data\uploaded_docs\test.pdf"

chunks = load_and_chunk_pdf(file_path)

# print first 3 chunks to see output
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk)
# test_setup.py

import fitz          
import faiss
import streamlit
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama

print("✅ PyMuPDF:", fitz.__version__)
print("✅ FAISS: OK")
print("✅ Streamlit:", streamlit.__version__)
print("✅ Loading embedding model... (may take 1-2 mins first time)")

model = SentenceTransformer("all-MiniLM-L6-v2")
test_embedding = model.encode(["hello world"])
print("✅ Sentence Transformers: OK, embedding shape:", test_embedding.shape)

response = ollama.chat(model="phi", messages=[{"role": "user", "content": "Say OK"}])
print("✅ Ollama + Phi:", response["message"]["content"])
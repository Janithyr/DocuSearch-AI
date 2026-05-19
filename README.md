DocuSearch AI
=============

A local, offline-first, privacy-first PDF Q&A app that turns your documents into
an interactive knowledge base. Upload a PDF, the app chunks and embeds the text,
stores vectors in FAISS, and uses a local Ollama model (Phi) to answer questions
based only on your document content.

Why DocuSearch AI
-----------------
- Local and private: no cloud uploads, no external APIs.
- Fast semantic search: FAISS indexes for quick retrieval.
- Clear answers: responses are grounded in the document context.
- Clean UI: focused chat experience built with Streamlit.

How It Works
------------
1) Extract text from your PDF.
2) Split into overlapping chunks.
3) Create embeddings with SentenceTransformers.
4) Store vectors in FAISS for similarity search.
5) Send the most relevant chunks to a local LLM (Phi via Ollama).

Features
--------
- PDF upload and auto-indexing
- Semantic search over document chunks
- Local LLM answers (Ollama + Phi)
- Offline-first workflow

Getting Started
---------------
1) Install dependencies
	pip install -r requirements.txt

2) Install and start Ollama
	- Download from https://ollama.ai
	- Run: ollama pull phi

3) Run the app
	streamlit run app.py

Usage
-----
- Upload a PDF.
- Click "Process Document" to index it.
- Ask questions in the chat box.

Project Structure
-----------------
- app.py: Streamlit UI and workflow
- src/pdf_loader.py: PDF text extraction and chunking
- src/embeddings.py: Embedding generation
- src/vector_store.py: FAISS index storage and search
- src/llm_handler.py: Ollama + Phi request handling

Planned Updates
---------------
- Multi-document collections and source-aware answers
- Inline citations (chunk references and page numbers)
- Better PDF parsing for tables and images
- Configurable chunk sizes and top-k retrieval
- Chat history export and session management
- Model selection from the UI

Notes
-----
- The FAISS index and uploaded files are generated at runtime.
- For best results, use clean, text-based PDFs.

License
-------
Choose a license (MIT recommended) before publishing.

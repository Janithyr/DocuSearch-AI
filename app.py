# app.py

import streamlit as st
import os
from src.pdf_loader import load_and_chunk_pdf
from src.embeddings import get_embeddings, get_query_embedding
from src.vector_store import save_vector_store, load_vector_store, search_similar_chunks
from src.llm_handler import get_answer

st.set_page_config(
    page_title="DocuSearch AI",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
    background: #080810;
    color: #e2e2f0;
}

/* ── Hide streamlit UI ── */
#MainMenu, footer, header, [data-testid="stSidebar"],
[data-testid="collapsedControl"] { display: none !important; }

/* ── Top navbar ── */
.navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 0 24px;
    border-bottom: 1px solid #13132a;
    margin-bottom: 36px;
}
.navbar-brand {
    display: flex; align-items: center; gap: 10px;
}
.navbar-icon {
    width: 34px; height: 34px; border-radius: 9px;
    background: linear-gradient(135deg, #3b5bdb, #7048e8);
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
}
.navbar-name {
    font-size: 1rem; font-weight: 700; color: #e2e2f0;
    letter-spacing: -0.3px;
}
.navbar-tag {
    font-size: 0.7rem; padding: 3px 9px; border-radius: 20px;
    background: #13132a; color: #7048e8;
    font-weight: 600; letter-spacing: 0.3px;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 48px 0 40px;
}
.hero-badge {
    display: inline-block;
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 1px; text-transform: uppercase;
    color: #7048e8; background: #13132a;
    border: 1px solid #2a1f5a;
    padding: 5px 14px; border-radius: 20px;
    margin-bottom: 20px;
}
.hero-title {
    font-size: 2.6rem; font-weight: 700;
    color: #e2e2f0; line-height: 1.15;
    letter-spacing: -1px; margin-bottom: 14px;
}
.hero-title span {
    background: linear-gradient(135deg, #7048e8, #3b5bdb, #0ea5e9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 0.95rem; color: #4a4a6a;
    font-weight: 400; max-width: 420px;
    margin: 0 auto; line-height: 1.6;
}

/* ── Upload card ── */
.upload-card {
    background: #0d0d1a;
    border: 1px solid #1a1a2e;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 20px;
}
.card-label {
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.8px; text-transform: uppercase;
    color: #4a4a6a; margin-bottom: 14px;
}

/* ── File badge ── */
.file-badge {
    display: flex; align-items: center; gap: 12px;
    background: #080810; border: 1px solid #1a1a2e;
    border-radius: 12px; padding: 12px 14px; margin-top: 14px;
}
.file-icon {
    width: 38px; height: 38px; border-radius: 9px;
    background: #1e3a8a;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; flex-shrink: 0;
}
.file-name { font-size: 0.875rem; font-weight: 500; color: #e2e2f0; }
.file-meta { font-size: 0.75rem; color: #4a4a6a; margin-top: 2px; }
.file-check {
    margin-left: auto; width: 22px; height: 22px;
    border-radius: 50%; background: #052e16;
    display: flex; align-items: center; justify-content: center;
    color: #34d399; font-size: 0.75rem; font-weight: 700;
}

/* ── Chat card ── */
.chat-card {
    background: #0d0d1a;
    border: 1px solid #1a1a2e;
    border-radius: 18px;
    padding: 20px 22px;
    margin-bottom: 16px;
    min-height: 220px;
}
.chat-empty {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 48px 0; color: #2a2a4a; text-align: center;
}
.chat-empty-icon { font-size: 2rem; margin-bottom: 10px; }
.chat-empty-text { font-size: 0.85rem; }

.msg-user {
    display: flex; justify-content: flex-end; margin: 12px 0;
}
.msg-user-bubble {
    background: linear-gradient(135deg, #1e3a8a, #3b1f8c);
    border-radius: 16px 16px 4px 16px;
    padding: 11px 16px; max-width: 78%;
    font-size: 0.875rem; color: #e2e2f0; line-height: 1.5;
}
.msg-bot { display: flex; gap: 10px; margin: 12px 0; align-items: flex-start; }
.msg-bot-avatar {
    width: 28px; height: 28px; border-radius: 7px; flex-shrink: 0;
    background: linear-gradient(135deg, #3b5bdb, #7048e8);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; color: white; font-weight: 700;
}
.msg-bot-bubble {
    background: #12121e; border: 1px solid #1a1a2e;
    border-radius: 4px 16px 16px 16px;
    padding: 11px 16px; max-width: 78%;
    font-size: 0.875rem; color: #c8c8e0; line-height: 1.6;
}

/* ── Status card ── */
.status-card {
    background: #0d0d1a;
    border: 1px solid #1a1a2e;
    border-radius: 18px;
    padding: 22px 28px;
    margin-top: 8px;
}
.status-title {
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.8px; text-transform: uppercase;
    color: #4a4a6a; margin-bottom: 16px;
}
.status-row {
    display: flex; gap: 12px;
}
.stat-box {
    flex: 1; background: #080810;
    border: 1px solid #1a1a2e; border-radius: 12px;
    padding: 14px; text-align: center;
}
.stat-num {
    font-size: 1.5rem; font-weight: 700;
    color: #7048e8; letter-spacing: -0.5px;
}
.stat-label {
    font-size: 0.72rem; color: #4a4a6a;
    margin-top: 3px; font-weight: 500;
}
.status-indicator {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 16px;
}
.dot-green {
    width: 8px; height: 8px; border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 6px #34d399;
}
.dot-gray {
    width: 8px; height: 8px; border-radius: 50%;
    background: #2a2a4a;
}
.status-text { font-size: 0.85rem; color: #c8c8e0; font-weight: 500; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #3b5bdb, #7048e8) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 11px 20px !important;
    font-size: 0.875rem !important; font-weight: 500 !important;
    width: 100% !important; transition: opacity 0.2s !important;
    margin-top: 4px !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] section {
    background: #080810 !important;
    border: 1.5px dashed #1e1e38 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="stFileUploaderDropzone"] { background: transparent !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { color: #3a3a5a !important; }

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #0d0d1a !important;
    border: 1px solid #1a1a2e !important;
    border-radius: 12px !important;
    margin-top: 10px !important;
    box-shadow: 0 0 0 1px rgba(112, 72, 232, 0.2) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #e2e2f0 !important;
    font-size: 0.9rem !important;
    min-height: 44px !important;
}

/* ── Mobile polish ── */
@media (max-width: 640px) {
    .hero { padding: 32px 0 28px; }
    .hero-title { font-size: 2.1rem; }
    .upload-card, .chat-card, .status-card { padding: 18px; }
    .status-row { flex-direction: column; }
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #7048e8 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e1e38; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────
for key, val in {
    "pdf_processed": False,
    "index": None,
    "chunks": None,
    "chat_history": [],
    "doc_name": "",
    "chunk_count": 0
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Navbar ───────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">
        <div class="navbar-icon">🔍</div>
        <div class="navbar-name">DocuSearch AI</div>
    </div>
    <div class="navbar-tag">HOSTED · API</div>
</div>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────
if not st.session_state.pdf_processed:
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">AI Document Intelligence</div>
        <div class="hero-title">Ask anything from<br><span>your documents</span></div>
        <div class="hero-sub">Upload a PDF and get instant answers — powered by hosted AI</div>
    </div>
    """, unsafe_allow_html=True)

# ── Upload card ──────────────────────────────────────────────
st.markdown('<div class="upload-card"><div class="card-label">📤 Upload Document</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    label_visibility="collapsed"
)

if uploaded_file:
    st.markdown(f"""
    <div class="file-badge">
        <div class="file-icon">📘</div>
        <div>
            <div class="file-name">{uploaded_file.name}</div>
            <div class="file-meta">PDF Document</div>
        </div>
        <div class="file-check">✓</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:14px"></div>', unsafe_allow_html=True)

    if st.button("⚡  Process Document"):
        temp_path = os.path.join(BASE_DIR, "data", "uploaded_docs", "temp.pdf")
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Reading and indexing your document..."):
            chunks = load_and_chunk_pdf(temp_path)
            embeddings = get_embeddings(chunks)
            save_vector_store(embeddings, chunks)
            index, chunks = load_vector_store()
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.pdf_processed = True
            st.session_state.doc_name = uploaded_file.name
            st.session_state.chunk_count = len(chunks)

        st.success(f"✅ Document indexed — {len(chunks)} chunks ready!")
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ── Chat card ────────────────────────────────────────────────
if st.session_state.pdf_processed:
    st.markdown('<div class="chat-card"><div class="card-label">💬 Chat</div>', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""
        <div class="chat-empty">
            <div class="chat-empty-icon">💬</div>
            <div class="chat-empty-text">Ask your first question below</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        chat_html = ""
        for chat in st.session_state.chat_history:
            chat_html += f"""
            <div class="msg-user">
                <div class="msg-user-bubble">{chat["question"]}</div>
            </div>
            <div class="msg-bot">
                <div class="msg-bot-avatar">AI</div>
                <div class="msg-bot-bubble">{chat["answer"]}</div>
            </div>"""
        st.markdown(chat_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Chat input ───────────────────────────────────────────────
if st.session_state.pdf_processed:
    st.markdown('<div class="card-label" style="margin-top:6px">✍️ Type your question</div>', unsafe_allow_html=True)
    query = st.chat_input("Ask anything about your document...")
    if query:
        with st.spinner("Thinking..."):
            qe = get_query_embedding(query)
            rc = search_similar_chunks(qe, st.session_state.index, st.session_state.chunks)
            try:
                answer = get_answer(query, rc)
            except Exception as exc:
                st.error(str(exc))
                st.stop()
        st.session_state.chat_history.append({"question": query, "answer": answer})
        st.rerun()

# ── Document status (bottom) ─────────────────────────────────
st.markdown('<div class="status-card"><div class="status-title">📊 Document Status</div>', unsafe_allow_html=True)

if st.session_state.pdf_processed:
    st.markdown(f"""
    <div class="status-indicator">
        <div class="dot-green"></div>
        <div class="status-text">{st.session_state.doc_name} — Ready</div>
    </div>
    <div class="status-row">
        <div class="stat-box">
            <div class="stat-num">{st.session_state.chunk_count}</div>
            <div class="stat-label">Chunks</div>
        </div>
        <div class="stat-box">
            <div class="stat-num">{len(st.session_state.chat_history)}</div>
            <div class="stat-label">Questions Asked</div>
        </div>
        <div class="stat-box">
            <div class="stat-num">384</div>
            <div class="stat-label">Vector Dims</div>
        </div>
        <div class="stat-box">
            <div class="stat-num">Groq</div>
            <div class="stat-label">LLM Model</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-indicator">
        <div class="dot-gray"></div>
        <div class="status-text" style="color:#3a3a5a">No document loaded</div>
    </div>
    <div class="status-row">
        <div class="stat-box"><div class="stat-num" style="color:#2a2a4a">—</div><div class="stat-label">Chunks</div></div>
        <div class="stat-box"><div class="stat-num" style="color:#2a2a4a">—</div><div class="stat-label">Questions Asked</div></div>
        <div class="stat-box"><div class="stat-num" style="color:#2a2a4a">—</div><div class="stat-label">Vector Dims</div></div>
        <div class="stat-box"><div class="stat-num" style="color:#2a2a4a">Groq</div><div class="stat-label">LLM Model</div></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
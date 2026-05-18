"""
Day 9 – Script 03: Streamlit Project App
=========================================
Full Streamlit UI for the RAG Q&A project.
Includes:
  - Chat interface with conversation history
  - Source document display
  - Latency and interaction metrics in sidebar
  - File upload to extend the knowledge base

Run:
    streamlit run 03_app.py

Requirements:
    pip install streamlit langchain langchain-community langchain-huggingface
                transformers sentence-transformers chromadb accelerate
    (Run 01_data_prep.py first to populate chroma_db/)
"""

import pathlib
import sys
import streamlit as st

# ── Page configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Q&A Bot – Day 9 Project",
    page_icon="🔬",
    layout="wide",
)

# ── Dependency check ──────────────────────────────────────────────
if not pathlib.Path("./chroma_db").exists():
    st.error(
        "ChromaDB not found! Please run `python 01_data_prep.py` first to ingest the corpus.",
        icon="⚠️",
    )
    st.stop()

# ── Lazy imports (keep startup fast) ─────────────────────────────
from rag_core import RAGSystem   # noqa: E402

# ── Load RAG system (cached) ─────────────────────────────────────
@st.cache_resource(show_spinner="Loading RAG system (first run may take ~60s)…")
def get_rag():
    return RAGSystem(persist_dir="./chroma_db", k=3)

rag = get_rag()

# ── Session state ─────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of (question, answer, latency_ms, sources)

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Session Metrics")
    st.metric("Total queries", st.session_state.total_queries)

    if st.session_state.history:
        latencies = [h[2] for h in st.session_state.history]
        avg_lat   = sum(latencies) / len(latencies)
        st.metric("Avg latency", f"{avg_lat:.0f} ms")

    st.divider()
    st.markdown("**Corpus Stats**")
    n_docs = rag.vectorstore._collection.count()
    st.info(f"📚 {n_docs} chunks indexed in ChromaDB")

    st.divider()
    if st.button("🗑️ Clear history"):
        st.session_state.history = []
        st.session_state.total_queries = 0
        st.rerun()

    st.divider()
    st.markdown("**Sample Questions**")
    samples = [
        "What is machine learning?",
        "Who invented the transformer?",
        "What is RAG?",
        "What are AI ethics concerns?",
    ]
    for s in samples:
        st.markdown(f"- *{s}*")

# ── Main UI ───────────────────────────────────────────────────────
st.title("🔬 RAG Q&A Bot")
st.caption("Ask questions about the AI/ML corpus · Powered by Flan-T5 + ChromaDB")

# Display history
for question, answer, latency_ms, sources in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.caption(f"⏱ {latency_ms:.0f}ms")
        with st.expander("📎 Source chunks"):
            for i, src in enumerate(sources, 1):
                st.write(f"**[{i}]** {src[:200]}")

# Chat input
question = st.chat_input("Ask a question about the corpus…")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            result = rag.ask(question)

        answer     = result["answer"]
        latency_ms = result["latency_ms"]
        sources    = result["sources"]

        st.markdown(answer)
        st.caption(f"⏱ {latency_ms:.0f}ms · {len(sources)} source chunks retrieved")

        with st.expander("📎 Source chunks"):
            for i, src in enumerate(sources, 1):
                st.write(f"**[{i}]** {src[:200]}")

    st.session_state.history.append((question, answer, latency_ms, sources))
    st.session_state.total_queries += 1

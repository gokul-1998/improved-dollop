"""
Day 7 – Script 01: Document Summarization
==========================================
Demonstrates:
  - HuggingFace BART-based abstractive summarization
  - Handling long documents with map-reduce chunking
  - Comparing summaries at different max_length settings

Run:
    python 01_summarization.py

Requirements:
    pip install transformers langchain langchain-community accelerate
"""

import time
from transformers import pipeline as hf_pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

print("=" * 60)
print("Day 7 – Script 01: Document Summarization")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# 1. Load summarization model
# ─────────────────────────────────────────────────────────────────
print("\n[1] Loading facebook/bart-large-cnn (~1.6GB, downloads once)...")
summarizer = hf_pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    max_length=130,
    min_length=30,
    do_sample=False,
)
print("    Model loaded ✓")

# ─────────────────────────────────────────────────────────────────
# 2. Short document summarization
# ─────────────────────────────────────────────────────────────────
ARTICLE = """
LangChain is an open-source Python framework designed to simplify the
development of applications powered by large language models. It provides
modular building blocks — including prompt templates, memory systems, chains,
and agent runtimes — that developers can compose to build sophisticated AI
pipelines. LangChain integrates with dozens of LLM providers including
HuggingFace, OpenAI, Anthropic, and Cohere. Its LangChain Expression Language
(LCEL) enables declarative, composable workflows using a pipe syntax. The
framework also supports RAG (Retrieval-Augmented Generation) by integrating
with vector stores like ChromaDB, FAISS, and Pinecone. LangSmith, the
companion observability platform, provides tracing and evaluation dashboards
for LangChain applications in production.
"""

print("\n[2] Summarizing a single article...")
print(f"  Input length: {len(ARTICLE.split())} words")
t0     = time.perf_counter()
result = summarizer(ARTICLE.strip())
ms     = (time.perf_counter() - t0) * 1000

summary = result[0]["summary_text"]
print(f"  Summary     : {summary}")
print(f"  Output words: {len(summary.split())}")
print(f"  Latency     : {ms:.0f}ms")

# ─────────────────────────────────────────────────────────────────
# 3. Effect of max_length on summary quality
# ─────────────────────────────────────────────────────────────────
print("\n[3] Effect of max_length on summary")
for mlen in [50, 100, 150]:
    summ = summarizer(ARTICLE.strip(), max_length=mlen, min_length=20)[0]["summary_text"]
    print(f"  max_length={mlen:3d} | {len(summ.split()):2d} words | {summ[:100]}...")

# ─────────────────────────────────────────────────────────────────
# 4. Map-Reduce for long documents
# ─────────────────────────────────────────────────────────────────
LONG_TEXT = (ARTICLE.strip() + " ") * 6   # Simulate a longer document

print(f"\n[4] Map-Reduce Summarization (long doc: {len(LONG_TEXT.split())} words)")
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
chunks   = splitter.split_text(LONG_TEXT)
print(f"  Chunks created: {len(chunks)}")

# Step 1: Summarise each chunk
chunk_summaries = []
for i, chunk in enumerate(chunks):
    s = summarizer(chunk, max_length=80, min_length=20)[0]["summary_text"]
    chunk_summaries.append(s)
    print(f"  Chunk {i+1} summary: {s[:80]}...")

# Step 2: Summarise the combined chunk summaries
combined = " ".join(chunk_summaries)
final    = summarizer(combined, max_length=130, min_length=30)[0]["summary_text"]
print(f"\n  Final map-reduce summary:\n  {final}")

print("\n✅  Script 01 complete.")

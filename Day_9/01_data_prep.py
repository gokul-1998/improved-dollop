"""
Day 9 – Script 01: Data Preparation
=====================================
Demonstrates:
  - Loading and cleaning a text corpus
  - Chunking with RecursiveCharacterTextSplitter
  - Comparing chunk sizes and their effect on retrieval
  - Ingesting chunks into a persisted ChromaDB vector store

Run:
    python 01_data_prep.py

Requirements:
    pip install langchain langchain-community langchain-huggingface
                sentence-transformers chromadb
"""

import re
import pathlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("=" * 60)
print("Day 9 – Script 01: Dataset Preparation")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# 1. Sample corpus (replace with your own document)
# ─────────────────────────────────────────────────────────────────
CORPUS = """
# Introduction to Artificial Intelligence

Artificial intelligence (AI) is the simulation of human intelligence
in machines that are programmed to think and learn. The term was coined
by John McCarthy in 1956. Modern AI research focuses on machine learning,
deep learning, and natural language processing.

## Machine Learning

Machine learning is a subset of AI that enables computers to learn from
data without being explicitly programmed. Key techniques include supervised
learning, unsupervised learning, and reinforcement learning.

In supervised learning, models are trained on labelled datasets. Common
algorithms include linear regression, decision trees, and support vector
machines. Neural networks are also widely used for supervised tasks.

## Deep Learning

Deep learning uses multi-layer neural networks to learn representations
of data. Convolutional neural networks (CNNs) excel at image tasks, while
recurrent neural networks (RNNs) and transformers handle sequential data
such as text.

The transformer architecture, introduced in the paper "Attention is All
You Need" (Vaswani et al., 2017), revolutionised natural language processing.
Models like BERT, GPT, and T5 are all based on transformers.

## Natural Language Processing

NLP is the branch of AI concerned with understanding and generating human
language. Key tasks include sentiment analysis, named entity recognition,
machine translation, summarisation, and question answering.

Large language models (LLMs) such as GPT-4, Claude, and Llama are trained
on massive text corpora and can perform many NLP tasks without task-specific
fine-tuning, using a paradigm called in-context learning or prompt engineering.

## Retrieval-Augmented Generation

RAG is a technique that combines retrieval systems with generative models.
Instead of relying solely on the LLM's parametric knowledge, RAG retrieves
relevant documents from a corpus and injects them into the prompt as context.
This reduces hallucinations and keeps answers grounded in real data.

## Ethical Considerations

AI raises important ethical questions around bias, fairness, transparency,
and accountability. Responsible AI development includes explainability,
robustness testing, privacy preservation, and alignment with human values.
"""

# Save to file so it can be loaded in other scripts
corpus_path = pathlib.Path("data/corpus.txt")
corpus_path.parent.mkdir(parents=True, exist_ok=True)
corpus_path.write_text(CORPUS, encoding="utf-8")
print(f"\n[1] Corpus saved → {corpus_path.resolve()}")
print(f"    Total characters: {len(CORPUS):,}")
print(f"    Total words     : {len(CORPUS.split()):,}")

# ─────────────────────────────────────────────────────────────────
# 2. Text cleaning
# ─────────────────────────────────────────────────────────────────
print("\n[2] Text Cleaning")

def clean_text(text: str) -> str:
    text = re.sub(r"#+\s*", "", text)           # remove markdown headings
    text = re.sub(r"\s+", " ", text)            # normalise whitespace
    text = text.strip()
    return text

cleaned = clean_text(CORPUS)
print(f"    Before: {len(CORPUS):,} chars")
print(f"    After : {len(cleaned):,} chars")

# ─────────────────────────────────────────────────────────────────
# 3. Chunking strategies comparison
# ─────────────────────────────────────────────────────────────────
print("\n[3] Chunking Strategy Comparison")

strategies = [
    ("Small  (200/20)",  200, 20),
    ("Medium (400/40)",  400, 40),
    ("Large  (700/70)",  700, 70),
]

for name, size, overlap in strategies:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(cleaned)
    avg    = sum(len(c) for c in chunks) // max(len(chunks), 1)
    print(f"  {name}: {len(chunks):3d} chunks | avg {avg:4d} chars/chunk")

# ─────────────────────────────────────────────────────────────────
# 4. Build and persist ChromaDB vector store
# ─────────────────────────────────────────────────────────────────
print("\n[4] Building ChromaDB vector store...")

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
chunks   = splitter.split_text(cleaned)

print(f"    Using {len(chunks)} chunks with all-MiniLM-L6-v2 embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

PERSIST_DIR = "./chroma_db"
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIR,
)
vectorstore.persist()

count = vectorstore._collection.count()
print(f"    Stored {count} vectors → {PERSIST_DIR}/")

# ─────────────────────────────────────────────────────────────────
# 5. Quick retrieval smoke test
# ─────────────────────────────────────────────────────────────────
print("\n[5] Retrieval Smoke Test")
test_queries = [
    "What is machine learning?",
    "Who introduced the transformer architecture?",
    "What is RAG?",
]

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
for q in test_queries:
    docs = retriever.get_relevant_documents(q)
    print(f"\n  Q: {q}")
    for i, d in enumerate(docs, 1):
        print(f"    [{i}] {d.page_content[:100]}...")

print("\n✅  Script 01 complete. Run 02_rag_core.py next.")

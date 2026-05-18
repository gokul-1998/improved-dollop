"""
Day 7 – Script 02: Q&A System with RAG
=======================================
Demonstrates:
  - ChromaDB in-memory vector store
  - HuggingFace embeddings (all-MiniLM-L6-v2)
  - RetrievalQA chain with Flan-T5
  - Querying with source document retrieval
  - Multiple retrieval strategies (k, similarity threshold)

Run:
    python 02_qa_system.py

Requirements:
    pip install langchain langchain-community langchain-huggingface
                transformers sentence-transformers chromadb accelerate
"""

import time
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline as hf_pipeline

print("=" * 60)
print("Day 7 – Script 02: Q&A System (RAG)")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# 1. Knowledge base
# ─────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = [
    "Python was created by Guido van Rossum and released in 1991.",
    "LangChain is an open-source framework for building LLM applications.",
    "FAISS stands for Facebook AI Similarity Search, a library for fast vector search.",
    "Transformers are neural networks based on the self-attention mechanism, introduced in 2017.",
    "ChromaDB is a free, embedded, open-source vector database.",
    "Sentence Transformers produce fixed-size dense embeddings for text sentences.",
    "RAG stands for Retrieval-Augmented Generation — retrieve docs, then generate an answer.",
    "Streamlit is a Python library for building interactive web apps for data and AI.",
    "Cosine similarity measures the angle between two vectors; range is -1 to +1.",
    "HuggingFace hosts thousands of free pre-trained models for NLP, vision, and audio.",
]

# ─────────────────────────────────────────────────────────────────
# 2. Build vector store
# ─────────────────────────────────────────────────────────────────
print("\n[1] Building ChromaDB vector store with sentence-transformers embeddings...")
embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_texts(KNOWLEDGE_BASE, embeddings)
print(f"    Indexed {len(KNOWLEDGE_BASE)} documents ✓")

# ─────────────────────────────────────────────────────────────────
# 3. Load LLM
# ─────────────────────────────────────────────────────────────────
print("\n[2] Loading Flan-T5-base...")
pipe = hf_pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=200)
llm  = HuggingFacePipeline(pipeline=pipe)
print("    Model loaded ✓")

# ─────────────────────────────────────────────────────────────────
# 4. Custom QA prompt
# ─────────────────────────────────────────────────────────────────
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the following context to answer the question accurately.
If the context doesn't contain the answer, say "I don't know."

Context:
{context}

Question: {question}
Answer:""",
)

# ─────────────────────────────────────────────────────────────────
# 5. Build QA chain with k=3 retrieval
# ─────────────────────────────────────────────────────────────────
print("\n[3] Building RetrievalQA chain (k=3)...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT},
)

# ─────────────────────────────────────────────────────────────────
# 6. Run test queries
# ─────────────────────────────────────────────────────────────────
TEST_QUESTIONS = [
    "Who created Python and when?",
    "What is RAG?",
    "What does FAISS stand for?",
    "What is Streamlit used for?",
    "What is the capital of France?",   # Not in KB — should say "I don't know"
]

print("\n[4] Running Q&A queries...\n")
for question in TEST_QUESTIONS:
    t0     = time.perf_counter()
    result = qa_chain({"query": question})
    ms     = (time.perf_counter() - t0) * 1000

    answer  = result["result"]
    sources = [d.page_content for d in result["source_documents"]]

    print(f"  Q: {question}")
    print(f"  A: {answer}")
    print(f"  Retrieved sources ({ms:.0f}ms):")
    for s in sources:
        print(f"    • {s[:90]}")
    print()

# ─────────────────────────────────────────────────────────────────
# 7. Compare retrieval with different k values
# ─────────────────────────────────────────────────────────────────
print("[5] Retrieval sensitivity — effect of k")
query = "Tell me about vector search and embeddings."
for k in [1, 2, 4]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs      = retriever.get_relevant_documents(query)
    print(f"  k={k}: {len(docs)} docs retrieved — top: '{docs[0].page_content[:70]}...'")

print("\n✅  Script 02 complete.")

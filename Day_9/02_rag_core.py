"""
Day 9 – Script 02: RAG Core System
====================================
Demonstrates:
  - RAGSystem class with persistent ChromaDB
  - Retriever + Flan-T5 + custom prompt
  - CLI interface for testing the pipeline
  - Logging each interaction to JSONL

Run:
    python 02_rag_core.py

Requirements:
    pip install langchain langchain-community langchain-huggingface
                transformers sentence-transformers chromadb accelerate
    (Run 01_data_prep.py first to populate chroma_db/)
"""

import json
import time
import datetime
import pathlib
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline as hf_pipeline

print("=" * 60)
print("Day 9 – Script 02: RAG Core System")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# RAGSystem class
# ─────────────────────────────────────────────────────────────────
class RAGSystem:
    """
    Reusable RAG pipeline:
      - Loads ChromaDB vector store from disk
      - Uses all-MiniLM-L6-v2 for embeddings
      - Uses Flan-T5-base for generation
      - Logs every interaction to rag_logs.jsonl
    """

    LOG_PATH = pathlib.Path("rag_logs.jsonl")

    def __init__(self, persist_dir: str = "./chroma_db", k: int = 3):
        print("\n  Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        print("  Loading ChromaDB from disk...")
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
        )
        n = self.vectorstore._collection.count()
        print(f"  Loaded {n} vectors from {persist_dir}/")

        print("  Loading Flan-T5-base LLM...")
        pipe = hf_pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=256,
        )
        llm = HuggingFacePipeline(pipeline=pipe)

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Answer the question using only the context below.\n"
                "If the answer is not in the context, say 'I don't know.'\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n"
                "Answer:"
            ),
        )

        self.chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": k}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        print("  RAGSystem ready ✓\n")

    def ask(self, question: str) -> dict:
        t0     = time.perf_counter()
        result = self.chain({"query": question})
        ms     = (time.perf_counter() - t0) * 1000

        answer  = result["result"]
        sources = [d.page_content for d in result["source_documents"]]

        self._log(question, answer, ms, sources)
        return {"answer": answer, "sources": sources, "latency_ms": round(ms, 1)}

    def _log(self, question: str, answer: str, ms: float, sources: list[str]):
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "question":  question,
            "answer":    answer,
            "latency_ms": round(ms, 1),
            "sources":   [s[:80] for s in sources],
        }
        with self.LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


# ─────────────────────────────────────────────────────────────────
# Run demo queries
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Check ChromaDB exists
    if not pathlib.Path("./chroma_db").exists():
        print("⚠️  ChromaDB not found. Run 01_data_prep.py first!")
        print("   python 01_data_prep.py")
        exit(1)

    rag = RAGSystem(persist_dir="./chroma_db", k=3)

    TEST_QUESTIONS = [
        "What is artificial intelligence?",
        "Who invented the transformer architecture?",
        "What is RAG and how does it help?",
        "What are the ethical considerations in AI?",
        "What is the capital of France?",   # Out-of-scope — should say I don't know
    ]

    print("=" * 60)
    print("Running test queries against the RAG pipeline")
    print("=" * 60)

    for q in TEST_QUESTIONS:
        print(f"\n  Q: {q}")
        result = rag.ask(q)
        print(f"  A: {result['answer']}")
        print(f"  ⏱  {result['latency_ms']}ms | {len(result['sources'])} source chunks")
        for i, s in enumerate(result["sources"], 1):
            print(f"     [{i}] {s[:80]}...")

    # Summary
    print(f"\n  Logged {len(TEST_QUESTIONS)} interactions → {RAGSystem.LOG_PATH.resolve()}")
    print("\n✅  Script 02 complete. Run `streamlit run 03_app.py` for the UI.")

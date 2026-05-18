# Day 9 – Project Phase 1: Design, Dataset & Development

Day 9 is your first full project day. You will choose a **problem statement**, prepare your **dataset**, architect a **RAG or chatbot solution** using the free stack you have learned, and write production-quality code in **Colab or VS Code**.

---

## 1. Part A — Problem Statement & Architecture Design

### 1.1 Choosing a Problem

Pick one of the following problem archetypes (or propose your own):

| Archetype | Example Project | Key Components |
|---|---|---|
| **Document Q&A** | FAQ Bot over a company PDF / Wikipedia dump | RAG + ChromaDB + Flan-T5 |
| **Customer Support Chatbot** | Answer queries from a product manual | Retrieval + Memory + Streamlit |
| **Summarization Pipeline** | Research paper → bullet-point summary | BART + LangChain map-reduce |
| **Code Helper** | Explain Python errors from a codebase | Embeddings + Code-specific LLM |

### 1.2 Architecture Diagram

Decide which architecture fits your problem:

```
── RAG Architecture ──────────────────────────────────────────────

User Query
    │
    ▼
[Embedding Model: all-MiniLM-L6-v2]
    │
    ▼
[Vector Store: ChromaDB / FAISS]   ←── [Document Corpus]
    │  top-k chunks
    ▼
[LLM: Flan-T5 / BART]
    │
    ▼
Answer + Source Citations


── Chatbot Architecture ──────────────────────────────────────────

User Message
    │
    ▼
[ConversationBufferMemory]
    │
    ▼
[LangChain ConversationChain]
    │
    ▼
[LLM: Flan-T5]
    │
    ▼
Response (with history)
```

### 1.3 Design Document Template

Create a `design.md` in your project folder with:

```markdown
# Project: [Your Title]

## Problem Statement
[2-3 sentences: what problem are you solving and for whom?]

## Dataset
- Source: [URL / file / generated]
- Format: [PDF / CSV / plain text / JSON]
- Size: [number of documents / tokens]
- Preprocessing: [chunking strategy, cleaning steps]

## Architecture
- Embedding model: [model name]
- Vector store: [ChromaDB / FAISS]
- LLM: [model name]
- Memory type: [none / buffer / summary]
- UI: [Streamlit / CLI]

## Evaluation Plan
- Metric 1: [ROUGE-L / latency / answer accuracy]
- Metric 2: [hallucination rate]
- Test queries: [list 5 sample queries]
```

---

## 2. Part B — Dataset Preparation

### 2.1 Loading and Cleaning Text Data

```python
import pathlib
import re

def load_text_file(path: str) -> str:
    """Load a plain text or markdown file."""
    return pathlib.Path(path).read_text(encoding="utf-8")

def clean_text(text: str) -> str:
    """Basic text cleaning pipeline."""
    text = re.sub(r"\s+", " ", text)           # normalise whitespace
    text = re.sub(r"[^\x20-\x7E\n]", "", text) # remove non-ASCII
    text = text.strip()
    return text

raw   = load_text_file("my_corpus.txt")
clean = clean_text(raw)
print(f"Characters: {len(clean):,}")
```

### 2.2 Chunking Strategies

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

# Strategy 1: Character-based (fast, good default)
char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# Strategy 2: Token-aware (respects model token limits)
token_splitter = SentenceTransformersTokenTextSplitter(
    model_name="all-MiniLM-L6-v2",
    chunk_overlap=25,
)

char_chunks  = char_splitter.split_text(clean)
token_chunks = token_splitter.split_text(clean)

print(f"Char chunks : {len(char_chunks)}  avg={sum(len(c) for c in char_chunks)//len(char_chunks)} chars")
print(f"Token chunks: {len(token_chunks)}")
```

### 2.3 Building the Vector Store

```python
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Persist to disk so you don't re-embed on every run
vectorstore = Chroma.from_texts(
    texts=char_chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)
vectorstore.persist()
print(f"Stored {vectorstore._collection.count()} chunks")
```

---

## 3. Part C — Core Application Development

### 3.1 Project File Structure

```
Day_9_Project/
├── data/
│   └── corpus.txt          # Your raw dataset
├── chroma_db/              # Persisted vector store (auto-created)
├── design.md               # Architecture document
├── 01_data_prep.py         # Dataset loading, cleaning, chunking, ingestion
├── 02_rag_core.py          # RAG pipeline (retriever + LLM + chain)
├── 03_app.py               # Streamlit UI
├── requirements.txt
└── README.md
```

### 3.2 RAG Core Module

```python
# 02_rag_core.py
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline as hf_pipeline

class RAGSystem:
    def __init__(self, persist_dir: str = "./chroma_db"):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
        pipe     = hf_pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
        llm      = HuggingFacePipeline(pipeline=pipe)
        self.chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )

    def ask(self, question: str) -> dict:
        result  = self.chain({"query": question})
        sources = [d.page_content[:120] for d in result["source_documents"]]
        return {"answer": result["result"], "sources": sources}


if __name__ == "__main__":
    rag = RAGSystem()
    out = rag.ask("What is the main topic of the corpus?")
    print("Answer:", out["answer"])
    print("Sources:")
    for s in out["sources"]:
        print(f"  - {s}")
```

### 3.3 Streamlit Project App

```python
# 03_app.py
import streamlit as st
from rag_core import RAGSystem

st.set_page_config(page_title="Project RAG Bot", page_icon="🤖")
st.title("🤖 Project: RAG Q&A Bot")

@st.cache_resource
def load_rag():
    return RAGSystem()

rag = load_rag()

if "history" not in st.session_state:
    st.session_state.history = []

for q, a in st.session_state.history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    st.divider()

question = st.text_input("Ask a question about the corpus:")
if st.button("Ask") and question:
    with st.spinner("Retrieving and generating..."):
        result = rag.ask(question)
    st.session_state.history.append((question, result["answer"]))
    st.markdown(f"**Answer:** {result['answer']}")
    with st.expander("Source chunks"):
        for s in result["sources"]:
            st.write(f"• {s}")
```

---

## 4. Hands-On Scripts

| File | What It Demonstrates |
|---|---|
| `01_data_prep.py` | Load, clean, chunk text; ingest into persisted ChromaDB |
| `02_rag_core.py` | Full RAGSystem class — retriever + Flan-T5 + chain |
| `03_app.py` | Streamlit project app with conversation history and source display |

```bash
cd Day_9
python 01_data_prep.py          # Ingest corpus into ChromaDB
python 02_rag_core.py           # Test RAG pipeline in CLI
streamlit run 03_app.py         # Launch Streamlit UI
```

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **Problem Statement** | Define clearly: who needs what, using which data source |
| **Chunking** | Chunk size 300–600 chars with 10% overlap is a good default |
| **Persistent ChromaDB** | Use `persist_directory` so embeddings survive restarts |
| **RAGSystem class** | Encapsulate retriever + LLM into a reusable object |
| **Design Doc** | Written architecture forces you to think before you code |

---

## Further Reading

- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [ChromaDB Persistence](https://docs.trychroma.com/usage-guide#initiating-a-persistent-chroma-client)
- [Flan-T5 on HuggingFace](https://huggingface.co/google/flan-t5-base)
- [Google Colab Free GPU](https://colab.research.google.com/)

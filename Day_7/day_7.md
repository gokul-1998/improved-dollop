# Day 7 – GenAI Applications: Summarization, Q&A & Chatbot with Streamlit

Today you build **three production-style GenAI applications** — a document summariser, a Q&A system, and a stateful chatbot — all powered by free LLMs and wrapped in a **Streamlit** UI. You will also touch on FastAPI for backend APIs and session-based memory.

---

## Technology Stack (100% Free)

| Layer | Technology |
|---|---|
| **LLM** | `facebook/bart-large-cnn` or `google/flan-t5-base` via HuggingFace |
| **Orchestration** | LangChain (LCEL, Memory, RetrievalQA) |
| **Embeddings** | `all-MiniLM-L6-v2` via sentence-transformers |
| **Vector Store** | ChromaDB (in-memory) |
| **UI** | Streamlit |
| **Backend API** | FastAPI (optional) |

```bash
pip install streamlit langchain langchain-community langchain-huggingface \
            transformers sentence-transformers chromadb fastapi uvicorn
```

---

## 1. Part A — Document Summarization

### Extractive vs Abstractive

| Type | How It Works | Model |
|---|---|---|
| **Extractive** | Selects key sentences from original | TextRank, BERT-Extractive |
| **Abstractive** | Generates new text capturing meaning | BART, T5, Pegasus |

### HuggingFace Summarization Pipeline

```python
from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    max_length=130, min_length=30, do_sample=False,
)
result = summarizer("LangChain is an open-source framework ...")
print(result[0]["summary_text"])
```

### Map-Reduce for Long Documents

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks   = splitter.create_documents([long_text])
chain    = load_summarize_chain(llm, chain_type="map_reduce")
summary  = chain.run(chunks)
```

### Streamlit Summarizer UI

```python
import streamlit as st
from transformers import pipeline

st.title("Document Summarizer")

@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()
text  = st.text_area("Paste document:", height=250)
mlen  = st.slider("Max length", 50, 250, 130)

if st.button("Summarize") and text.strip():
    with st.spinner("Summarizing..."):
        out = summarizer(text, max_length=mlen, min_length=30)
    st.write(out[0]["summary_text"])
```

---

## 2. Part B — Q&A System (RAG-based)

### Architecture

```
Question → Embed → ChromaDB Retrieval → Context → LLM → Answer
```

### Building the Pipeline

```python
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_texts([
    "Python was created by Guido van Rossum in 1991.",
    "LangChain is an open-source LLM application framework.",
    "ChromaDB is a free embedded vector database.",
], embeddings)

hf_pipe  = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=200)
llm      = HuggingFacePipeline(pipeline=hf_pipe)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
)

result = qa_chain({"query": "Who created Python?"})
print("Answer:", result["result"])
```

---

## 3. Part C — Stateful Chatbot & FastAPI Backend

### Streamlit Session Memory Chatbot

```python
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

st.title("GenAI Chatbot")

if "memory"   not in st.session_state:
    st.session_state.memory   = ConversationBufferMemory()
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_llm():
    pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=200)
    return HuggingFacePipeline(pipeline=pipe)

llm = get_llm()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    chain    = ConversationChain(llm=llm, memory=st.session_state.memory)
    response = chain.predict(input=user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
```

### FastAPI Backend Pattern

```python
# backend.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app  = FastAPI()
pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=200)

class ChatRequest(BaseModel):
    message: str
    history: list[str] = []

@app.post("/chat")
def chat(req: ChatRequest):
    prompt = "\n".join(req.history + [f"Human: {req.message}", "AI:"])
    return {"response": pipe(prompt)[0]["generated_text"]}
```

Run:
```bash
uvicorn backend:app --reload        # Terminal 1
streamlit run 03_chatbot_streamlit.py  # Terminal 2
```

---

## 4. Hands-On Scripts

| File | What It Demonstrates |
|---|---|
| `01_summarization.py` | BART summarizer, map-reduce for long docs, Streamlit UI |
| `02_qa_system.py` | ChromaDB + HuggingFace embeddings + RetrievalQA + Streamlit UI |
| `03_chatbot_streamlit.py` | Session memory chatbot, FastAPI backend pattern |

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **Summarization** | BART/T5 for abstractive; map-reduce for docs > context window |
| **Q&A with RAG** | Embed docs in ChromaDB → retrieve top-k → feed as context |
| **Streamlit Session State** | Persists memory and messages across re-runs |
| **FastAPI** | Separates LLM from UI; enables multiple frontends |
| **Authentication** | `streamlit-authenticator` or FastAPI `OAuth2PasswordBearer` |

---

## Further Reading

- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain RetrievalQA](https://python.langchain.com/docs/use_cases/question_answering/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [HuggingFace Spaces (free hosting)](https://huggingface.co/spaces)

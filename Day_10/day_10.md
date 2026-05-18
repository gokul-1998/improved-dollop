# Day 10 – Project Phase 2: Testing, Deployment & Final Presentation

Day 10 is your finish line. You will **test** your project rigorously, **deploy** it to a free hosting platform (HuggingFace Spaces or Streamlit Cloud), and **present** your work. We also cover the most important GenAI interview questions you should know.

---

## 1. Part A — Testing Your GenAI Application

### 1.1 Testing Strategy

| Layer | What to Test | Tool |
|---|---|---|
| **Unit** | Individual functions (chunking, cleaning, guardrails) | `pytest` |
| **Integration** | RAG pipeline end-to-end | `pytest` + assertions |
| **Quality** | LLM output quality on a test set | ROUGE, LLM-judge |
| **Regression** | Ensure changes don't break existing queries | CI test suite |

### 1.2 Writing Unit Tests with pytest

```python
# tests/test_data_prep.py
import pytest
from data_prep import clean_text, chunk_text   # your modules

def test_clean_text_removes_extra_whitespace():
    raw    = "Hello   World\n\n\nHow  are  you"
    result = clean_text(raw)
    assert "  " not in result

def test_clean_text_strips_non_ascii():
    raw    = "Café au lait"
    result = clean_text(raw)
    assert "é" not in result

def test_chunk_text_size():
    text   = "word " * 1000
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
    assert all(len(c) <= 220 for c in chunks)   # allow slight overflow at boundaries
    assert len(chunks) > 1

def test_rag_returns_answer():
    from rag_core import RAGSystem
    rag    = RAGSystem()
    result = rag.ask("What is this project about?")
    assert "answer" in result
    assert len(result["answer"]) > 5
```

Run:
```bash
pytest tests/ -v
```

### 1.3 Quality Evaluation on a Test Set

```python
# evaluate.py
from rouge_score import rouge_scorer
from rag_core import RAGSystem

# Define ground-truth QA pairs
TEST_SET = [
    {
        "question": "Who created Python?",
        "reference": "Python was created by Guido van Rossum.",
    },
    {
        "question": "What is LangChain?",
        "reference": "LangChain is an open-source framework for building LLM applications.",
    },
]

rag    = RAGSystem()
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
scores = []

for item in TEST_SET:
    result   = rag.ask(item["question"])
    hypothesis = result["answer"]
    reference  = item["reference"]
    score    = scorer.score(reference, hypothesis)["rougeL"].fmeasure
    scores.append(score)
    print(f"Q: {item['question']}")
    print(f"  Predicted : {hypothesis}")
    print(f"  Reference : {reference}")
    print(f"  ROUGE-L   : {score:.3f}\n")

print(f"Mean ROUGE-L: {sum(scores)/len(scores):.3f}")
```

---

## 2. Part B — Deployment to Free Platforms

### 2.1 HuggingFace Spaces (Recommended)

HuggingFace Spaces hosts Streamlit apps for **free** with 2 CPU / 16 GB RAM.

**Step-by-step:**

```bash
# 1. Create a free account at https://huggingface.co/

# 2. Create a new Space (choose Streamlit SDK)
#    https://huggingface.co/new-space

# 3. Clone the Space repo
git clone https://huggingface.co/spaces/<username>/<space-name>
cd <space-name>

# 4. Copy your project files into the repo
cp -r ../Day_9_Project/* .

# 5. Create requirements.txt
cat requirements.txt   # make sure all deps are listed

# 6. Push to HuggingFace
git add .
git commit -m "Initial deploy"
git push
```

Your app will be live at `https://<username>-<space-name>.hf.space`

### 2.2 Streamlit Community Cloud

```bash
# 1. Push your code to a public GitHub repo

# 2. Go to https://share.streamlit.io/
#    Connect GitHub → Select repo → Select main file (app.py)

# 3. Add secrets (API keys) in the dashboard under Settings > Secrets
#    HUGGINGFACEHUB_API_TOKEN = "hf_..."
```

### 2.3 requirements.txt for Deployment

```
langchain>=0.2.0
langchain-community>=0.2.0
langchain-huggingface>=0.0.3
transformers>=4.40.0
sentence-transformers>=2.7.0
chromadb>=0.5.0
streamlit>=1.35.0
rouge-score>=0.1.2
```

### 2.4 Optimising for Free Tier

| Issue | Fix |
|---|---|
| Model too large for free RAM | Use `google/flan-t5-small` (300MB) instead of base |
| Slow first load | Pre-embed corpus offline; push `chroma_db/` to the repo |
| Environment variable secrets | Use `st.secrets` (Streamlit) or HF Space secrets |
| Cold start delay | Add `@st.cache_resource` on model loading |

```python
# Safe secret loading for both local and deployed
import os
import streamlit as st

def get_secret(key: str, default: str = "") -> str:
    """Read from st.secrets (deployed) or env vars (local)."""
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)
```

---

## 3. Part C — Final Presentation & GenAI Interview Questions

### 3.1 Presentation Structure (10 minutes)

| Section | Time | Content |
|---|---|---|
| **Problem** | 1 min | What problem, who benefits |
| **Architecture** | 2 min | Diagram: data → embeddings → retrieval → LLM → UI |
| **Demo** | 4 min | Live demo with 3-5 test queries |
| **Evaluation** | 2 min | ROUGE scores, latency, limitations |
| **Learnings** | 1 min | What you would do differently |

### 3.2 GenAI Interview Questions — Must Know

**Fundamentals**

| Question | Key Answer |
|---|---|
| What is a transformer? | Attention-based architecture; encoder-decoder or decoder-only |
| What is attention? | Weighted sum of values based on query-key similarity |
| What is a prompt? | Input text that conditions the LLM's output |
| What is temperature in LLMs? | Controls randomness; 0=deterministic, >1=creative |
| What is top-p sampling? | Sample from smallest set of tokens summing to probability p |

**RAG & Embeddings**

| Question | Key Answer |
|---|---|
| What is RAG? | Retrieve relevant docs → feed as context → LLM answers |
| Why RAG over fine-tuning? | No GPU/training cost; knowledge stays up-to-date |
| What is cosine similarity? | Measures angle between vectors; range −1 to +1 |
| What is chunking? | Splitting docs into smaller pieces to fit context window |
| FAISS vs ChromaDB? | FAISS=library (manual metadata); Chroma=embedded DB (built-in metadata) |

**LangChain & Agents**

| Question | Key Answer |
|---|---|
| What is a chain? | Sequential composition of prompt → LLM → output |
| What is memory in LangChain? | Persists conversation history between turns |
| What is an agent? | LLM that decides which tools to call in a loop |
| What is ReAct? | Reason-Act pattern: Thought → Action → Observation → repeat |
| What is LCEL? | LangChain Expression Language; uses `\|` to chain components |

**Evaluation & Safety**

| Question | Key Answer |
|---|---|
| What is hallucination? | Model generates plausible but factually incorrect text |
| How do you detect hallucinations? | Check if claims are grounded in retrieved source docs |
| What is a guardrail? | Input/output filter to block harmful or sensitive content |
| What metrics evaluate summarization? | ROUGE-1/2/L, BERTScore |
| What is LLM-as-judge? | Use a second LLM to score responses on a rubric |

### 3.3 Common Interview Pitfalls

```
❌ "I used ChatGPT/OpenAI" → ✅ Explain the free stack you used and why
❌ "It just works" → ✅ Quantify: latency, ROUGE score, number of test queries
❌ Vague architecture → ✅ Draw the data flow from input to output
❌ No failure cases → ✅ Mention what breaks and how you would fix it
```

---

## 4. Hands-On Scripts

| File | What It Demonstrates |
|---|---|
| `01_testing.py` | pytest unit tests + quality evaluation on test set |
| `02_deployment_prep.py` | Requirements check, secret loading, ChromaDB pre-embedding |
| `03_interview_prep.py` | Interactive quiz to self-test GenAI interview answers |

```bash
cd Day_10
pytest 01_testing.py -v
python 02_deployment_prep.py
python 03_interview_prep.py
```

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **pytest** | Write unit + integration tests before deploying |
| **ROUGE-L** | Quick, free quality metric for generated answers |
| **HuggingFace Spaces** | Best free hosting for Streamlit apps; 2 CPU / 16 GB RAM |
| **Streamlit Cloud** | Connects directly to GitHub; easiest for quick deploys |
| **Secrets** | Use `st.secrets` / HF Space secrets — never commit API keys |
| **Interview Prep** | Know RAG, attention, hallucination, ROUGE, ReAct cold |

---

## Further Reading

- [HuggingFace Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Streamlit Community Cloud](https://streamlit.io/cloud)
- [pytest Documentation](https://docs.pytest.org/)
- [GenAI Interview Questions (GitHub)](https://github.com/aishwaryanr/awesome-generative-ai-guide)
- [Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)

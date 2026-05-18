# Day 8 – Optimization & Evaluation: Prompt Tuning, Guardrails & Observability

Building a GenAI application is only half the job. Day 8 focuses on making it **reliable, safe, and fast** — through systematic prompt tuning, evaluation frameworks, latency optimisation, basic guardrails for safety, and open-source observability tools.

---

## 1. Part A — Prompt Tuning & Evaluation Techniques

### 1.1 Why Prompt Tuning Matters

The same model can produce wildly different outputs depending on how you phrase the prompt. Prompt tuning is the process of **systematically improving prompts** to achieve consistent, high-quality responses.

### 1.2 Core Prompt Engineering Strategies

| Technique | Description | When to Use |
|---|---|---|
| **Zero-shot** | No examples; rely on model knowledge | Simple, well-defined tasks |
| **Few-shot** | 2–5 input-output examples in prompt | Classification, formatting |
| **Chain-of-Thought (CoT)** | "Think step by step" instruction | Reasoning, math |
| **Self-Consistency** | Sample multiple outputs, vote on best | Critical answers |
| **Instruction Formatting** | Clear role + task + constraints | All use cases |

```python
# Zero-shot
zero_shot = "Classify the sentiment of: 'This movie was fantastic!'"

# Few-shot
few_shot = """
Classify sentiment as Positive/Negative/Neutral.

Text: "I love this product!" -> Positive
Text: "It was okay." -> Neutral
Text: "Terrible experience." -> Negative
Text: "This movie was fantastic!" ->
"""

# Chain-of-Thought
cot = """
Solve step by step:
A store sells apples for $0.50 each. Alice buys 7 apples. How much does she pay?
Step 1: ...
"""
```

### 1.3 Automated Prompt Evaluation

Evaluate prompts using a **rubric-based approach** with a judge LLM:

```python
from transformers import pipeline

judge = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=50)

def evaluate_response(question: str, response: str) -> dict:
    """Use an LLM as a judge to score a response."""
    rubric = f"""
Rate the following response on a scale of 1-5 for:
1. Correctness  2. Completeness  3. Clarity

Question: {question}
Response: {response}

Output only: Correctness=X, Completeness=X, Clarity=X
"""
    raw = judge(rubric)[0]["generated_text"]
    scores = {}
    for part in raw.split(","):
        if "=" in part:
            k, v = part.strip().split("=")
            scores[k.strip()] = int(v.strip())
    return scores

score = evaluate_response(
    "What is gradient descent?",
    "Gradient descent minimises a loss function by iteratively moving in the direction of steepest descent."
)
print(score)
```

### 1.4 BLEU, ROUGE, BERTScore

```python
# pip install rouge-score bert-score nltk
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

reference = "Gradient descent is an optimisation algorithm that minimises a function iteratively."
hypothesis = "Gradient descent iteratively reduces a loss function."

scores = scorer.score(reference, hypothesis)
for key, val in scores.items():
    print(f"{key}: P={val.precision:.3f}  R={val.recall:.3f}  F1={val.fmeasure:.3f}")
```

---

## 2. Part B — Latency Optimisation & Hallucination Detection

### 2.1 Latency Bottlenecks and Fixes

| Bottleneck | Fix |
|---|---|
| Model loading on every request | `@st.cache_resource` / singleton pattern |
| Long context window | Reduce chunk size; use summarisation memory |
| Slow tokenisation | Batch requests; use fast tokenisers |
| Retrieval over large corpus | Use IVF FAISS index; limit `k` |
| Network latency (cloud LLMs) | Use local model or cache responses |

```python
import time
from functools import lru_cache

# Cache model output for repeated identical queries
@lru_cache(maxsize=256)
def cached_generate(prompt: str) -> str:
    return pipe(prompt)[0]["generated_text"]

# Measure latency
start = time.perf_counter()
result = cached_generate("What is machine learning?")
print(f"Latency: {(time.perf_counter()-start)*1000:.1f}ms")
```

### 2.2 Hallucination Detection (Basic)

Hallucination = the model confidently states something false. Basic detection strategies:

```python
def detect_hallucination(response: str, source_docs: list[str]) -> dict:
    """
    Check if key claims in the response are grounded in source documents.
    Returns a basic grounding score.
    """
    response_words = set(response.lower().split())
    matched = 0
    for doc in source_docs:
        doc_words = set(doc.lower().split())
        overlap   = response_words & doc_words
        matched   = max(matched, len(overlap))

    # Rough grounding ratio
    grounding = matched / max(len(response_words), 1)
    is_grounded = grounding > 0.25

    return {
        "grounding_score": round(grounding, 3),
        "is_grounded": is_grounded,
        "verdict": "GROUNDED" if is_grounded else "POSSIBLE HALLUCINATION",
    }

response = "Python was created by Guido van Rossum in 1991."
sources  = ["Python is a programming language. Guido van Rossum created Python."]
print(detect_hallucination(response, sources))
```

### 2.3 Guardrails — Input and Output Safety

```python
import re

# Input guardrail — reject harmful prompts
BLOCKED_PATTERNS = [
    r"\b(hack|exploit|jailbreak)\b",
    r"\b(password|credit card|ssn)\b",
]

def input_guardrail(prompt: str) -> tuple[bool, str]:
    """Returns (is_safe, reason)."""
    for pat in BLOCKED_PATTERNS:
        if re.search(pat, prompt, re.IGNORECASE):
            return False, f"Blocked pattern: '{pat}'"
    return True, "OK"

# Output guardrail — detect PII leakage
PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",            # SSN
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
]

def output_guardrail(text: str) -> tuple[str, list]:
    """Redact PII from model output."""
    issues = []
    for pat in PII_PATTERNS:
        matches = re.findall(pat, text)
        if matches:
            issues.extend(matches)
            text = re.sub(pat, "[REDACTED]", text)
    return text, issues
```

---

## 3. Part C — Observability with Open-Source Tools

### 3.1 Why Observability?

Observability answers: *What is my LLM doing, and is it doing it well?*

| Signal | What It Measures |
|---|---|
| **Traces** | End-to-end latency per chain step |
| **Metrics** | Token usage, cache hit rate, error rate |
| **Logs** | Full prompt and response for debugging |
| **Evaluations** | Quality scores (ROUGE, BERTScore, LLM-judge) |

### 3.2 LangSmith (Free Tier) — LangChain Tracing

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]     = "your-free-langsmith-key"
os.environ["LANGCHAIN_PROJECT"]     = "day8-optimization"

# All LangChain chain calls are now automatically traced
from langchain.chains import LLMChain
# ... rest of your chain code — tracing is automatic
```

### 3.3 Minimal Custom Logger

```python
import json, datetime, pathlib

LOG_FILE = pathlib.Path("llm_logs.jsonl")

def log_interaction(prompt: str, response: str, latency_ms: float, scores: dict):
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "prompt":    prompt,
        "response":  response,
        "latency_ms": latency_ms,
        "scores":    scores,
    }
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")

def load_logs() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    with LOG_FILE.open() as f:
        return [json.loads(line) for line in f]
```

### 3.4 Observability Tools Landscape

| Tool | Type | Cost |
|---|---|---|
| **LangSmith** | LangChain-native tracing | Free tier |
| **Weights & Biases (W&B)** | Experiment tracking | Free tier |
| **Phoenix (Arize)** | LLM observability UI | Open-source |
| **Prometheus + Grafana** | General metrics/dashboards | Open-source |
| **OpenTelemetry** | Vendor-neutral tracing standard | Open-source |

---

## 4. Hands-On Scripts

| File | What It Demonstrates |
|---|---|
| `01_prompt_evaluation.py` | Few-shot prompts, LLM-as-judge scoring, ROUGE metrics |
| `02_latency_guardrails.py` | Latency measurement, caching, hallucination detection, input/output guards |
| `03_observability.py` | Custom JSONL logger, LangSmith tracing setup, metrics dashboard |

```bash
cd Day_8
python 01_prompt_evaluation.py
python 02_latency_guardrails.py
python 03_observability.py
```

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **Prompt Tuning** | Few-shot + CoT dramatically improves output quality |
| **LLM-as-Judge** | Use a second model to evaluate responses on a rubric |
| **ROUGE / BERTScore** | Quantitative metrics for summarization and generation quality |
| **Hallucination Detection** | Check if claims are grounded in retrieved source documents |
| **Guardrails** | Block harmful inputs; redact PII from outputs |
| **Caching** | `lru_cache` or Redis for repeated queries saves seconds of latency |
| **Observability** | LangSmith / Phoenix for traces; JSONL for lightweight local logging |

---

## Further Reading

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Arize Phoenix (open-source observability)](https://github.com/Arize-ai/phoenix)
- [ROUGE Score Library](https://github.com/google-research/google-research/tree/master/rouge)
- [BERTScore Paper](https://arxiv.org/abs/1904.09675)
- [NEMO Guardrails (NVIDIA)](https://github.com/NVIDIA/NeMo-Guardrails)
- [Chain-of-Thought Prompting Paper](https://arxiv.org/abs/2201.11903)

"""
Day 8 – Script 03: Observability & LLM Logging
===============================================
Demonstrates:
  - JSONL-based structured logging for every LLM call
  - Computing and reporting aggregate metrics
  - LangSmith tracing setup (environment variables)
  - Simple Streamlit metrics dashboard (CLI fallback shown)

Run:
    python 03_observability.py

Requirements:
    pip install transformers accelerate
"""

import json
import time
import datetime
import pathlib
import statistics
from transformers import pipeline as hf_pipeline

print("=" * 60)
print("Day 8 – Script 03: Observability & LLM Logging")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# Load LLM
# ─────────────────────────────────────────────────────────────────
print("\n[0] Loading Flan-T5-base...")
pipe = hf_pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=150)
print("    Model loaded ✓")

# ─────────────────────────────────────────────────────────────────
# 1. Structured JSONL logger
# ─────────────────────────────────────────────────────────────────
LOG_PATH = pathlib.Path("llm_logs.jsonl")

def log_interaction(
    prompt: str,
    response: str,
    latency_ms: float,
    model: str = "flan-t5-base",
    extra: dict = None,
):
    """Append one interaction to the JSONL log file."""
    entry = {
        "timestamp":  datetime.datetime.utcnow().isoformat() + "Z",
        "model":      model,
        "prompt":     prompt,
        "response":   response,
        "latency_ms": round(latency_ms, 1),
        "input_len":  len(prompt.split()),
        "output_len": len(response.split()),
    }
    if extra:
        entry.update(extra)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def load_logs() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    with LOG_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

print(f"\n[1] Logging to: {LOG_PATH.resolve()}")

# ─────────────────────────────────────────────────────────────────
# 2. Run 10 instrumented queries
# ─────────────────────────────────────────────────────────────────
QUERIES = [
    "What is machine learning?",
    "Define gradient descent in one sentence.",
    "What is RAG in NLP?",
    "Explain attention mechanism briefly.",
    "What is a vector database?",
    "What is cosine similarity?",
    "Define tokenisation.",
    "What is fine-tuning an LLM?",
    "What does FAISS stand for?",
    "What is LangChain?",
]

print(f"\n[2] Running {len(QUERIES)} instrumented queries...\n")
for i, q in enumerate(QUERIES, 1):
    t0       = time.perf_counter()
    response = pipe(q)[0]["generated_text"].strip()
    ms       = (time.perf_counter() - t0) * 1000
    log_interaction(prompt=q, response=response, latency_ms=ms)
    print(f"  [{i:02d}] {ms:5.0f}ms | {q[:45]:<46} → {response[:50]}")

# ─────────────────────────────────────────────────────────────────
# 3. Aggregate metrics report
# ─────────────────────────────────────────────────────────────────
print("\n[3] Aggregate Metrics from Logs")
logs     = load_logs()
latency  = [e["latency_ms"] for e in logs]
in_lens  = [e["input_len"]  for e in logs]
out_lens = [e["output_len"] for e in logs]

print(f"\n  Total interactions : {len(logs)}")
print(f"  {'Metric':<25} {'Min':>8} {'Max':>8} {'Mean':>8} {'P50':>8} {'P90':>8}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

def report_metric(name, values):
    srt = sorted(values)
    p50 = srt[len(srt)//2]
    p90 = srt[int(len(srt)*0.9)]
    print(
        f"  {name:<25} {min(values):>8.1f} {max(values):>8.1f} "
        f"{statistics.mean(values):>8.1f} {p50:>8.1f} {p90:>8.1f}"
    )

report_metric("Latency (ms)",   latency)
report_metric("Input tokens",   in_lens)
report_metric("Output tokens",  out_lens)

# ─────────────────────────────────────────────────────────────────
# 4. LangSmith setup snippet
# ─────────────────────────────────────────────────────────────────
print("\n[4] LangSmith Tracing Setup (environment variables)")
print("""
  To enable automatic tracing for all LangChain calls:

  import os
  os.environ["LANGCHAIN_TRACING_V2"] = "true"
  os.environ["LANGCHAIN_API_KEY"]     = "ls_..."   # From app.smith.langchain.com
  os.environ["LANGCHAIN_PROJECT"]     = "day8-demo"

  After that, every chain.run() / chain.invoke() call is automatically
  traced and visible in the LangSmith dashboard — no code changes needed.
""")

# ─────────────────────────────────────────────────────────────────
# 5. Observability tools landscape
# ─────────────────────────────────────────────────────────────────
print("[5] Open-Source Observability Tools\n")
tools = [
    ("LangSmith",          "LangChain-native tracing",    "Free tier"),
    ("Arize Phoenix",      "LLM observability UI",        "Open-source"),
    ("Weights & Biases",   "Experiment + model tracking", "Free tier"),
    ("OpenTelemetry",      "Vendor-neutral tracing",      "Open-source"),
    ("Prometheus+Grafana", "Metrics & dashboards",        "Open-source"),
]
print(f"  {'Tool':<25} {'Type':<30} {'Cost'}")
print(f"  {'-'*25} {'-'*30} {'-'*15}")
for name, typ, cost in tools:
    print(f"  {name:<25} {typ:<30} {cost}")

print(f"\n  Log file: {LOG_PATH.resolve()} ({LOG_PATH.stat().st_size} bytes)")
print("\n✅  Script 03 complete.")

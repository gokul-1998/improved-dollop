"""
Day 8 – Script 01: Prompt Evaluation & Quality Metrics
=======================================================
Demonstrates:
  - Zero-shot, few-shot, and Chain-of-Thought prompts
  - ROUGE-L scoring for generated text quality
  - LLM-as-judge evaluation rubric
  - A/B comparison of prompt strategies

Run:
    python 01_prompt_evaluation.py

Requirements:
    pip install transformers langchain-huggingface rouge-score accelerate
"""

import time
from transformers import pipeline as hf_pipeline
from rouge_score import rouge_scorer as rouge_lib

print("=" * 60)
print("Day 8 – Script 01: Prompt Evaluation & Quality Metrics")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────
print("\n[0] Loading models...")
generator = hf_pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=200)
print("    generator loaded ✓")

def generate(prompt: str) -> str:
    return generator(prompt)[0]["generated_text"].strip()

# ─────────────────────────────────────────────────────────────────
# 1. Prompt strategies side-by-side
# ─────────────────────────────────────────────────────────────────
print("\n[1] Prompt Strategy Comparison")

QUESTION = "What is gradient descent?"

prompts = {
    "zero_shot": QUESTION,
    "few_shot": (
        "Q: What is backpropagation?\n"
        "A: Backpropagation is an algorithm that computes gradients of the loss w.r.t weights using the chain rule.\n\n"
        f"Q: {QUESTION}\nA:"
    ),
    "chain_of_thought": (
        f"Think step by step to answer: {QUESTION}\n"
        "Step 1: What problem does it solve?\n"
        "Step 2: How does it work?\n"
        "Step 3: Give the final concise answer.\n"
        "Answer:"
    ),
}

responses = {}
for name, prompt in prompts.items():
    t0 = time.perf_counter()
    resp = generate(prompt)
    ms   = (time.perf_counter() - t0) * 1000
    responses[name] = resp
    print(f"\n  [{name}] ({ms:.0f}ms)")
    print(f"    {resp[:150]}...")

# ─────────────────────────────────────────────────────────────────
# 2. ROUGE scoring
# ─────────────────────────────────────────────────────────────────
print("\n[2] ROUGE-L Scoring vs Reference Answer")

REFERENCE = (
    "Gradient descent is an optimisation algorithm that minimises a loss "
    "function by iteratively moving parameters in the direction of steepest descent."
)
scorer = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

print(f"\n  Reference: {REFERENCE}\n")
print(f"  {'Strategy':<22} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8}")
print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}")

for name, resp in responses.items():
    s = scorer.score(REFERENCE, resp)
    print(
        f"  {name:<22} "
        f"{s['rouge1'].fmeasure:>8.3f} "
        f"{s['rouge2'].fmeasure:>8.3f} "
        f"{s['rougeL'].fmeasure:>8.3f}"
    )

# ─────────────────────────────────────────────────────────────────
# 3. LLM-as-Judge evaluation
# ─────────────────────────────────────────────────────────────────
print("\n[3] LLM-as-Judge (rubric-based scoring)")

def llm_judge(question: str, response: str) -> dict:
    """Score a response on correctness, completeness, and clarity (1-5)."""
    rubric_prompt = (
        f"Rate this response on a scale 1-5 for each criterion.\n"
        f"Question: {question}\n"
        f"Response: {response}\n\n"
        f"Respond in exactly this format:\n"
        f"Correctness: X\nCompleteness: X\nClarity: X"
    )
    raw = generate(rubric_prompt)
    scores = {}
    for line in raw.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip()
            if v.isdigit():
                scores[k] = int(v)
    return scores

for name, resp in list(responses.items())[:2]:   # judge first 2 to save time
    scores = llm_judge(QUESTION, resp)
    avg    = sum(scores.values()) / len(scores) if scores else 0
    print(f"\n  [{name}]")
    for k, v in scores.items():
        print(f"    {k:<15} : {'★' * v} ({v}/5)")
    print(f"    Average        : {avg:.1f}/5")

print("\n✅  Script 01 complete.")

"""
Day 10 – Script 03: GenAI Interview Prep Quiz
==============================================
Interactive CLI quiz to self-test your GenAI interview knowledge.
Covers: Transformers, RAG, LangChain, Embeddings, Evaluation, Safety.

Run:
    python 03_interview_prep.py

Requirements:
    No extra packages needed — uses only the Python standard library.
"""

import random
import sys

print("=" * 60)
print("Day 10 – Script 03: GenAI Interview Prep Quiz")
print("=" * 60)
print("Test your knowledge before your interview!\n")

# ─────────────────────────────────────────────────────────────────
# Question bank
# ─────────────────────────────────────────────────────────────────
QUESTIONS = [
    # ── Fundamentals ──────────────────────────────────────────────
    {
        "q": "What does LLM stand for?",
        "a": "Large Language Model",
        "hint": "Think about the scale of these models.",
        "topic": "Fundamentals",
    },
    {
        "q": "What is the key innovation in the transformer architecture?",
        "a": "Self-attention mechanism",
        "hint": "It allows each token to attend to every other token.",
        "topic": "Fundamentals",
    },
    {
        "q": "What does 'temperature' control in LLM generation?",
        "a": "Randomness/creativity of output; 0=deterministic, high=more random",
        "hint": "Higher value = more creative but less predictable.",
        "topic": "Fundamentals",
    },
    {
        "q": "What is the difference between encoder-only and decoder-only transformers?",
        "a": "Encoder-only (BERT) understands text; decoder-only (GPT) generates text",
        "hint": "BERT vs GPT",
        "topic": "Fundamentals",
    },
    {
        "q": "What is top-p (nucleus) sampling?",
        "a": "Sample from the smallest set of tokens whose cumulative probability >= p",
        "hint": "It limits the vocabulary dynamically.",
        "topic": "Fundamentals",
    },
    # ── RAG & Embeddings ──────────────────────────────────────────
    {
        "q": "What does RAG stand for?",
        "a": "Retrieval-Augmented Generation",
        "hint": "It combines two main capabilities.",
        "topic": "RAG",
    },
    {
        "q": "Why use RAG instead of fine-tuning?",
        "a": "No GPU/training cost; knowledge stays current; reduces hallucinations",
        "hint": "Think about cost, speed, and knowledge freshness.",
        "topic": "RAG",
    },
    {
        "q": "What is cosine similarity and what is its range?",
        "a": "Measures angle between two vectors; range -1 (opposite) to +1 (identical)",
        "hint": "For semantic similarity, scores >0.7 are typically 'similar'.",
        "topic": "RAG",
    },
    {
        "q": "What is chunking in RAG pipelines?",
        "a": "Splitting documents into smaller pieces to fit the LLM context window",
        "hint": "Too large = won't fit context; too small = loses context.",
        "topic": "RAG",
    },
    {
        "q": "What is the difference between FAISS and ChromaDB?",
        "a": "FAISS=library with manual metadata; ChromaDB=embedded DB with built-in metadata",
        "hint": "One is a raw library, the other is a full database.",
        "topic": "RAG",
    },
    # ── LangChain ─────────────────────────────────────────────────
    {
        "q": "What is a LangChain Chain?",
        "a": "Sequential composition of prompt → LLM → output parser",
        "hint": "LCEL uses the | operator to compose chains.",
        "topic": "LangChain",
    },
    {
        "q": "What does ConversationBufferMemory do?",
        "a": "Stores all conversation messages verbatim for the LLM to use as context",
        "hint": "Good for short sessions; grows without bound.",
        "topic": "LangChain",
    },
    {
        "q": "What is the ReAct pattern in agents?",
        "a": "Reason-Act loop: LLM alternates Thought → Action → Observation until done",
        "hint": "Think → Do → Observe → Repeat",
        "topic": "LangChain",
    },
    {
        "q": "What is LCEL?",
        "a": "LangChain Expression Language; uses | to chain prompt | llm | parser",
        "hint": "It's the modern way to compose LangChain components.",
        "topic": "LangChain",
    },
    # ── Evaluation & Safety ───────────────────────────────────────
    {
        "q": "What is hallucination in LLMs?",
        "a": "When the model generates plausible-sounding but factually incorrect text",
        "hint": "The model is confident but wrong.",
        "topic": "Evaluation",
    },
    {
        "q": "How do you detect hallucinations in RAG systems?",
        "a": "Check if the response claims are grounded in the retrieved source documents",
        "hint": "Compare response words to source document words.",
        "topic": "Evaluation",
    },
    {
        "q": "What is ROUGE-L used for?",
        "a": "Measuring quality of generated text by comparing longest common subsequences with reference",
        "hint": "It's a recall-precision-F1 metric for text generation.",
        "topic": "Evaluation",
    },
    {
        "q": "What is a guardrail in GenAI systems?",
        "a": "An input/output filter that blocks harmful prompts or redacts sensitive data",
        "hint": "Safety layer before and after the LLM.",
        "topic": "Evaluation",
    },
    {
        "q": "What is LLM-as-judge evaluation?",
        "a": "Using a second LLM to score responses on a rubric (correctness, clarity, etc.)",
        "hint": "A model evaluating another model's output.",
        "topic": "Evaluation",
    },
    # ── Deployment ────────────────────────────────────────────────
    {
        "q": "What is st.cache_resource used for in Streamlit?",
        "a": "Caching expensive resources like model objects across re-runs of the script",
        "hint": "Without it, the model reloads on every user interaction.",
        "topic": "Deployment",
    },
    {
        "q": "Name two free platforms to host a Streamlit GenAI app.",
        "a": "HuggingFace Spaces and Streamlit Community Cloud",
        "hint": "One is by HuggingFace, the other by Streamlit.",
        "topic": "Deployment",
    },
]


# ─────────────────────────────────────────────────────────────────
# Quiz runner
# ─────────────────────────────────────────────────────────────────
def run_quiz(questions: list[dict], n: int = 10) -> None:
    selected = random.sample(questions, min(n, len(questions)))
    correct  = 0
    skipped  = 0

    for i, item in enumerate(selected, 1):
        print(f"\n{'─'*60}")
        print(f"  [{item['topic']}] Q{i}/{len(selected)}")
        print(f"  {item['q']}")
        print()
        user = input("  Your answer (or 'h' for hint, 's' to skip): ").strip()

        if user.lower() == "h":
            print(f"  💡 Hint: {item['hint']}")
            user = input("  Your answer: ").strip()

        if user.lower() in ("s", "skip", ""):
            print(f"  ⏭  Skipped. Answer: {item['a']}")
            skipped += 1
            continue

        print(f"\n  ✅ Model answer: {item['a']}")
        grade = input("  Did you get it right? (y/n): ").strip().lower()
        if grade.startswith("y"):
            correct += 1
            print("  🎉 Great job!")
        else:
            print("  📚 Review this topic.")

    # Results
    attempted = len(selected) - skipped
    pct = (correct / attempted * 100) if attempted > 0 else 0
    print(f"\n{'='*60}")
    print(f"  Quiz Complete!")
    print(f"  Score    : {correct}/{attempted} ({pct:.0f}%)")
    print(f"  Skipped  : {skipped}")
    if pct >= 80:
        print("  Rating   : 🌟 Interview-ready!")
    elif pct >= 60:
        print("  Rating   : 📖 Good — review weak areas")
    else:
        print("  Rating   : 📚 More practice needed — reread the day_*.md files")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────
# Study mode — show all Q&A by topic
# ─────────────────────────────────────────────────────────────────
def study_mode(questions: list[dict]) -> None:
    topics = sorted(set(q["topic"] for q in questions))
    for topic in topics:
        print(f"\n{'─'*60}")
        print(f"  📚 {topic}")
        print(f"{'─'*60}")
        for item in questions:
            if item["topic"] == topic:
                print(f"\n  Q: {item['q']}")
                print(f"  A: {item['a']}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n  {len(QUESTIONS)} questions across 5 topics:")
    topics = sorted(set(q["topic"] for q in QUESTIONS))
    for t in topics:
        n = sum(1 for q in QUESTIONS if q["topic"] == t)
        print(f"    • {t} ({n} questions)")

    print("\n  Choose mode:")
    print("    1. Quiz mode  (interactive, randomly selected questions)")
    print("    2. Study mode (all Q&A shown by topic)")
    print("    3. Exit")

    choice = input("\n  Enter 1, 2, or 3: ").strip()

    if choice == "1":
        n_str = input(f"  How many questions? (1-{len(QUESTIONS)}, default 10): ").strip()
        n     = int(n_str) if n_str.isdigit() else 10
        run_quiz(QUESTIONS, n=n)
    elif choice == "2":
        study_mode(QUESTIONS)
    else:
        print("  Goodbye! Good luck with your interview. 🚀")

    print("\n✅  Script 03 complete.")

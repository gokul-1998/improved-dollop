"""
Day 10 – Script 01: Testing the Project
=========================================
Demonstrates:
  - Unit tests for data prep functions
  - Integration test for the full RAG pipeline
  - Quality evaluation on a ground-truth test set
  - ROUGE-L scoring of generated answers

Run:
    python 01_testing.py
    # or with pytest:
    pytest 01_testing.py -v

Requirements:
    pip install rouge-score langchain langchain-community langchain-huggingface
                transformers sentence-transformers chromadb accelerate
    (Run Day_9/01_data_prep.py first if chroma_db/ does not exist)
"""

import re
import sys
import pathlib
import unittest
import statistics

print("=" * 60)
print("Day 10 – Script 01: Testing the Project")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# Utility functions under test (mirrors Day_9 code)
# ─────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 400, chunk_overlap: int = 40) -> list[str]:
    """Simple character-based chunker."""
    chunks = []
    start  = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def grounding_score(response: str, sources: list[str]) -> float:
    resp_words = set(response.lower().split())
    best = 0
    for doc in sources:
        doc_words = set(doc.lower().split())
        best = max(best, len(resp_words & doc_words))
    return round(best / max(len(resp_words), 1), 3)


# ─────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────
class TestCleanText(unittest.TestCase):

    def test_removes_extra_whitespace(self):
        raw = "Hello   World\n\n\n How  are  you"
        self.assertNotIn("  ", clean_text(raw))

    def test_removes_markdown_headings(self):
        raw = "# Title\n## Subtitle\nContent here."
        result = clean_text(raw)
        self.assertNotIn("#", result)

    def test_strips_leading_trailing_whitespace(self):
        raw = "  hello world  "
        self.assertEqual(clean_text(raw), "hello world")

    def test_empty_string(self):
        self.assertEqual(clean_text(""), "")


class TestChunkText(unittest.TestCase):

    def test_chunk_count_is_positive(self):
        text   = "word " * 500
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)

    def test_no_chunk_exceeds_size_by_much(self):
        text   = "word " * 500
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
        for c in chunks:
            self.assertLessEqual(len(c), 210)

    def test_single_chunk_for_short_text(self):
        text   = "short text"
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        self.assertEqual(len(chunks), 1)


class TestGroundingScore(unittest.TestCase):

    def test_high_score_for_grounded_response(self):
        resp    = "Python was created by Guido van Rossum."
        sources = ["Guido van Rossum created Python programming language."]
        score   = grounding_score(resp, sources)
        self.assertGreater(score, 0.25)

    def test_low_score_for_hallucinated_response(self):
        resp    = "Python was invented by Ada Lovelace in 1843."
        sources = ["Guido van Rossum created Python programming language."]
        score   = grounding_score(resp, sources)
        self.assertLess(score, 0.4)

    def test_empty_sources(self):
        score = grounding_score("Some response", [])
        self.assertEqual(score, 0.0)


# ─────────────────────────────────────────────────────────────────
# Run unit tests
# ─────────────────────────────────────────────────────────────────
print("\n[1] Running Unit Tests...")
loader  = unittest.TestLoader()
suite   = unittest.TestSuite()
for cls in [TestCleanText, TestChunkText, TestGroundingScore]:
    suite.addTests(loader.loadTestsFromTestCase(cls))

runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
result = runner.run(suite)
print(f"\n  Passed: {result.testsRun - len(result.failures) - len(result.errors)}/{result.testsRun}")

# ─────────────────────────────────────────────────────────────────
# Integration test + quality evaluation
# ─────────────────────────────────────────────────────────────────
print("\n[2] Integration Test – RAG Pipeline Quality Evaluation")

CHROMA_DIR = pathlib.Path("../Day_9/chroma_db")
if not CHROMA_DIR.exists():
    print(f"  ⚠️  ChromaDB not found at {CHROMA_DIR}")
    print("     Run: python ../Day_9/01_data_prep.py")
    print("     Skipping integration tests.")
else:
    try:
        from rouge_score import rouge_scorer as rouge_lib
        sys.path.insert(0, str(pathlib.Path("../Day_9")))
        from rag_core import RAGSystem   # noqa: E402

        rag    = RAGSystem(persist_dir=str(CHROMA_DIR), k=3)
        scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)

        TEST_SET = [
            {
                "question":  "What is machine learning?",
                "reference": "Machine learning is a subset of AI that enables computers to learn from data.",
            },
            {
                "question":  "What is RAG?",
                "reference": "RAG is retrieval-augmented generation, combining retrieval with generative models.",
            },
            {
                "question":  "What is the transformer architecture?",
                "reference": "Transformers use self-attention and were introduced in the paper Attention is All You Need.",
            },
        ]

        rouge_scores = []
        ground_scores = []

        print(f"\n  {'Question':<45} {'ROUGE-L':>8} {'Grounding':>10}")
        print(f"  {'-'*45} {'-'*8} {'-'*10}")

        for item in TEST_SET:
            result    = rag.ask(item["question"])
            hypothesis = result["answer"]
            sources    = result["sources"]

            rouge_f1   = scorer.score(item["reference"], hypothesis)["rougeL"].fmeasure
            ground     = grounding_score(hypothesis, sources)
            rouge_scores.append(rouge_f1)
            ground_scores.append(ground)

            print(f"  {item['question'][:44]:<45} {rouge_f1:>8.3f} {ground:>10.3f}")

        print(f"\n  Mean ROUGE-L    : {statistics.mean(rouge_scores):.3f}")
        print(f"  Mean Grounding  : {statistics.mean(ground_scores):.3f}")

    except Exception as e:
        print(f"  Integration test error: {e}")

print("\n✅  Script 01 complete.")

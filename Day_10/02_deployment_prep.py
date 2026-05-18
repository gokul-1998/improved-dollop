"""
Day 10 – Script 02: Deployment Preparation
============================================
Demonstrates:
  - Verifying requirements for deployment
  - Pre-embedding corpus and packaging ChromaDB for hosting
  - Safe secret loading (st.secrets vs env vars)
  - HuggingFace Spaces and Streamlit Cloud checklists

Run:
    python 02_deployment_prep.py

Requirements:
    pip install langchain langchain-community langchain-huggingface
                sentence-transformers chromadb
"""

import os
import sys
import pathlib
import importlib
import subprocess

print("=" * 60)
print("Day 10 – Script 02: Deployment Preparation")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# 1. Dependency check
# ─────────────────────────────────────────────────────────────────
print("\n[1] Checking required packages...")

REQUIRED = [
    ("streamlit",             "streamlit"),
    ("langchain",             "langchain"),
    ("langchain_community",   "langchain-community"),
    ("langchain_huggingface", "langchain-huggingface"),
    ("transformers",          "transformers"),
    ("sentence_transformers", "sentence-transformers"),
    ("chromadb",              "chromadb"),
    ("rouge_score",           "rouge-score"),
]

all_ok = True
print(f"\n  {'Package':<30} {'Status'}")
print(f"  {'-'*30} {'-'*15}")
for module, pip_name in REQUIRED:
    try:
        importlib.import_module(module)
        print(f"  {pip_name:<30} ✅ installed")
    except ImportError:
        print(f"  {pip_name:<30} ❌ MISSING  →  pip install {pip_name}")
        all_ok = False

if all_ok:
    print("\n  All packages present ✓")
else:
    print("\n  ⚠️  Install missing packages before deploying.")

# ─────────────────────────────────────────────────────────────────
# 2. ChromaDB pre-embedding check
# ─────────────────────────────────────────────────────────────────
print("\n[2] ChromaDB readiness check...")

chroma_path = pathlib.Path("../Day_9/chroma_db")
if chroma_path.exists():
    size_mb = sum(f.stat().st_size for f in chroma_path.rglob("*") if f.is_file()) / 1_048_576
    print(f"  chroma_db/ found at {chroma_path.resolve()}")
    print(f"  Size: {size_mb:.1f} MB — OK to include in git repo (small enough)")
    print("  ✅ Pre-embedded — no embedding cost on deploy startup")
else:
    print("  ⚠️  chroma_db/ not found. Run: python ../Day_9/01_data_prep.py")

# ─────────────────────────────────────────────────────────────────
# 3. Safe secret loading utility
# ─────────────────────────────────────────────────────────────────
print("\n[3] Safe Secret Loading Pattern")

def get_secret(key: str, default: str = "") -> str:
    """
    Load a secret from st.secrets (Streamlit Cloud / HF Spaces) or
    fall back to environment variables (local dev).
    """
    try:
        import streamlit as st
        return st.secrets.get(key, os.environ.get(key, default))
    except Exception:
        return os.environ.get(key, default)

# Demo — show which source would be used
print("""
  Usage in your app:

    hf_token = get_secret("HUGGINGFACEHUB_API_TOKEN")
    api_key  = get_secret("OPENAI_API_KEY")

  Priority:
    1. st.secrets["KEY"]           ← Streamlit Cloud / HF Spaces settings
    2. os.environ["KEY"]           ← Local .env or shell export
    3. default=""                  ← Fails gracefully
""")

# ─────────────────────────────────────────────────────────────────
# 4. requirements.txt generator
# ─────────────────────────────────────────────────────────────────
print("[4] Generating requirements.txt for deployment...")

DEPLOY_REQS = """streamlit>=1.35.0
langchain>=0.2.0
langchain-community>=0.2.0
langchain-huggingface>=0.0.3
transformers>=4.40.0
sentence-transformers>=2.7.0
chromadb>=0.5.0
rouge-score>=0.1.2
accelerate>=0.27.0
"""

req_path = pathlib.Path("requirements.txt")
req_path.write_text(DEPLOY_REQS, encoding="utf-8")
print(f"  Written to {req_path.resolve()}")

# ─────────────────────────────────────────────────────────────────
# 5. Deployment checklist
# ─────────────────────────────────────────────────────────────────
print("\n[5] Deployment Checklist\n")

checklists = {
    "HuggingFace Spaces": [
        "Create free account at https://huggingface.co/",
        "New Space → SDK: Streamlit → Hardware: CPU Basic (free)",
        "Clone space repo:  git clone https://huggingface.co/spaces/<user>/<name>",
        "Copy project files + chroma_db/ into the repo",
        "Ensure requirements.txt is present and correct",
        "Push: git add . && git commit -m 'deploy' && git push",
        "Add secrets in Space Settings → Repository secrets",
    ],
    "Streamlit Community Cloud": [
        "Push project to a PUBLIC GitHub repository",
        "Go to https://share.streamlit.io/ and sign in with GitHub",
        "New app → select repo / branch / main file (03_app.py)",
        "Add secrets in App Settings → Secrets",
        "Click Deploy — live in ~2 minutes",
    ],
}

for platform, steps in checklists.items():
    print(f"  ── {platform} ──")
    for i, step in enumerate(steps, 1):
        print(f"    {i}. {step}")
    print()

# ─────────────────────────────────────────────────────────────────
# 6. Model size optimisation for free tier
# ─────────────────────────────────────────────────────────────────
print("[6] Model Size Optimisation for Free Hosting Tier\n")
models = [
    ("google/flan-t5-small",  "~300MB",  "Fast startup, good for free tier"),
    ("google/flan-t5-base",   "~990MB",  "Good quality, fits free tier (16GB RAM)"),
    ("google/flan-t5-large",  "~3GB",    "Better quality, may exceed free tier"),
    ("all-MiniLM-L6-v2",      "~90MB",   "Embedding model — always fits"),
]
print(f"  {'Model':<40} {'Size':<10} {'Notes'}")
print(f"  {'-'*40} {'-'*10} {'-'*35}")
for model, size, notes in models:
    print(f"  {model:<40} {size:<10} {notes}")

print("\n✅  Script 02 complete.")

"""
Day 6 – Script 02: Conversation Memory
=======================================
Demonstrates:
  - ConversationBufferMemory (stores all turns)
  - ConversationBufferWindowMemory (last k turns)
  - ConversationSummaryMemory (LLM-compressed history)
  - Multi-turn conversation simulation

Run:
    python 02_memory_conversation.py

Requirements:
    pip install langchain langchain-community langchain-huggingface transformers accelerate
"""

from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
)
from langchain.chains import ConversationChain
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline as hf_pipeline

print("=" * 60)
print("Day 6 – Script 02: Conversation Memory")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# Load LLM once
# ─────────────────────────────────────────────────────────────────
print("\n[0] Loading Flan-T5-base...")
pipe = hf_pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=150)
llm  = HuggingFacePipeline(pipeline=pipe)
print("    Model loaded ✓")

# ─────────────────────────────────────────────────────────────────
# 1. ConversationBufferMemory – stores everything
# ─────────────────────────────────────────────────────────────────
print("\n[1] ConversationBufferMemory (stores ALL messages)")

buffer_memory = ConversationBufferMemory()
conv_buffer   = ConversationChain(llm=llm, memory=buffer_memory, verbose=False)

turns = [
    "My name is Alice and I am studying NLP.",
    "What field am I studying?",
    "What are the main tasks in that field?",
]

for turn in turns:
    response = conv_buffer.predict(input=turn)
    print(f"  Human : {turn}")
    print(f"  AI    : {response}\n")

print(f"  Stored messages: {len(buffer_memory.chat_memory.messages)}")
print("  Full history:")
for msg in buffer_memory.chat_memory.messages:
    role = "Human" if msg.type == "human" else "AI"
    print(f"    [{role}] {msg.content[:80]}...")

# ─────────────────────────────────────────────────────────────────
# 2. ConversationBufferWindowMemory – last k turns
# ─────────────────────────────────────────────────────────────────
print("\n[2] ConversationBufferWindowMemory (last k=2 turns)")

window_memory = ConversationBufferWindowMemory(k=2)
conv_window   = ConversationChain(llm=llm, memory=window_memory, verbose=False)

for i, turn in enumerate(turns, 1):
    response = conv_window.predict(input=turn)
    hist     = window_memory.chat_memory.messages
    print(f"  Turn {i}: '{turn[:50]}...'  → messages in window: {len(hist)}")

# ─────────────────────────────────────────────────────────────────
# 3. ConversationSummaryMemory – LLM compresses history
# ─────────────────────────────────────────────────────────────────
print("\n[3] ConversationSummaryMemory (LLM-compressed)")

summary_memory = ConversationSummaryMemory(llm=llm)
conv_summary   = ConversationChain(llm=llm, memory=summary_memory, verbose=False)

for turn in turns[:2]:
    response = conv_summary.predict(input=turn)
    print(f"  Human  : {turn}")
    print(f"  AI     : {response}")
    print(f"  Summary: {summary_memory.buffer[:100]}...\n")

# ─────────────────────────────────────────────────────────────────
# 4. Manual memory inspection
# ─────────────────────────────────────────────────────────────────
print("\n[4] Memory Comparison Table")
print(f"  {'Type':<35} {'Messages stored':<18} {'Notes'}")
print(f"  {'-'*35} {'-'*18} {'-'*30}")
print(f"  {'ConversationBufferMemory':<35} {'All turns':<18} {'Good for short sessions'}")
print(f"  {'ConversationBufferWindowMemory(k=2)':<35} {'Last 2 turns':<18} {'Constant size'}")
print(f"  {'ConversationSummaryMemory':<35} {'LLM summary':<18} {'Best for long sessions'}")

print("\n✅  Script 02 complete.")

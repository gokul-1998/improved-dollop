"""
Day 7 – Script 03: Stateful Chatbot with Streamlit
===================================================
Demonstrates:
  - Streamlit session_state for persistent multi-turn memory
  - ConversationBufferMemory integrated with Streamlit
  - Chat UI with st.chat_message and st.chat_input
  - FastAPI backend pattern (shown as CLI simulation)

Run (Streamlit UI):
    streamlit run 03_chatbot_streamlit.py

Run (CLI demo without Streamlit):
    python 03_chatbot_streamlit.py --cli

Requirements:
    pip install streamlit langchain langchain-community langchain-huggingface
                transformers accelerate
"""

import sys

# ─────────────────────────────────────────────────────────────────
# CLI demo mode (no Streamlit required)
# ─────────────────────────────────────────────────────────────────
if "--cli" in sys.argv:
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationChain
    from langchain_huggingface import HuggingFacePipeline
    from transformers import pipeline as hf_pipeline

    print("=" * 60)
    print("Day 7 – Script 03: Chatbot (CLI mode)")
    print("Type 'quit' to exit. Type 'history' to see memory.")
    print("=" * 60)

    print("\nLoading Flan-T5-base...")
    pipe   = hf_pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=200)
    llm    = HuggingFacePipeline(pipeline=pipe)
    memory = ConversationBufferMemory()
    chain  = ConversationChain(llm=llm, memory=memory, verbose=False)
    print("Chatbot ready!\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "history":
            msgs = memory.chat_memory.messages
            print(f"  [{len(msgs)} messages in memory]")
            for m in msgs:
                role = "Human" if m.type == "human" else "AI"
                print(f"  [{role}] {m.content[:80]}")
            continue

        response = chain.predict(input=user_input)
        print(f"Bot: {response}\n")

    sys.exit(0)

# ─────────────────────────────────────────────────────────────────
# Streamlit Chatbot App
# ─────────────────────────────────────────────────────────────────
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline as hf_pipeline

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="GenAI Chatbot – Day 7",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 GenAI Chatbot")
st.caption("Powered by Flan-T5 (free, runs locally) · Session memory via LangChain")

# ── Session state initialisation ──────────────────────────────────
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your GenAI chatbot. Ask me anything!"}
    ]

# ── Load LLM (cached across re-runs) ─────────────────────────────
@st.cache_resource(show_spinner="Loading Flan-T5-base (first run only)…")
def get_llm():
    pipe = hf_pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=200,
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = get_llm()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Session Info")
    n_turns = len(st.session_state.messages)
    st.metric("Messages", n_turns)
    st.metric("Memory messages", len(st.session_state.memory.chat_memory.messages))

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Conversation cleared. Let's start fresh!"}
        ]
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
        st.rerun()

    st.markdown("---")
    st.markdown("**Tips:**\n- Try multi-turn conversations\n- Ask follow-up questions\n- Type 'summarise our chat' to test memory")

# ── Render chat history ───────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────
if user_input := st.chat_input("Type your message…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            chain    = ConversationChain(llm=llm, memory=st.session_state.memory)
            response = chain.predict(input=user_input)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

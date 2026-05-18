"""
Day 6 – Script 01: Chains & Prompt Templates
=============================================
Demonstrates:
  - PromptTemplate and ChatPromptTemplate basics
  - LLMChain with HuggingFace free LLM (Flan-T5)
  - LangChain Expression Language (LCEL) pipe operator
  - Sequential chains and output parsers

Run:
    python 01_chains_prompts.py

Requirements:
    pip install langchain langchain-community langchain-huggingface transformers accelerate
"""

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline as hf_pipeline

print("=" * 60)
print("Day 6 – Script 01: Chains & Prompt Templates")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# 1. Load a free HuggingFace LLM (Flan-T5-base)
# ─────────────────────────────────────────────────────────────────
print("\n[1] Loading Flan-T5-base (downloads ~1GB on first run)...")
pipe = hf_pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=200,
)
llm = HuggingFacePipeline(pipeline=pipe)
print("    Model loaded ✓")

# ─────────────────────────────────────────────────────────────────
# 2. PromptTemplate basics
# ─────────────────────────────────────────────────────────────────
print("\n[2] PromptTemplate basics")
template = PromptTemplate(
    input_variables=["topic", "level"],
    template="Explain {topic} to a {level} student in exactly 2 bullet points.",
)

for topic, level in [("neural networks", "beginner"), ("attention mechanism", "advanced")]:
    prompt_text = template.format(topic=topic, level=level)
    print(f"\n  Prompt : {prompt_text}")
    response = llm.invoke(prompt_text)
    print(f"  Response: {response}")

# ─────────────────────────────────────────────────────────────────
# 3. ChatPromptTemplate
# ─────────────────────────────────────────────────────────────────
print("\n[3] ChatPromptTemplate")
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI tutor specialising in {subject}."),
    ("human",  "{question}"),
])
messages = chat_prompt.format_messages(
    subject="machine learning",
    question="What is gradient descent?"
)
print(f"  System  : {messages[0].content}")
print(f"  Human   : {messages[1].content}")
response = llm.invoke(messages[1].content)  # Flan-T5 is text2text, pass human msg
print(f"  Response: {response}")

# ─────────────────────────────────────────────────────────────────
# 4. LCEL – pipe operator chain
# ─────────────────────────────────────────────────────────────────
print("\n[4] LCEL Pipe Operator (prompt | llm | parser)")

simple_prompt = PromptTemplate.from_template("What is {concept} in one sentence?")
chain = simple_prompt | llm | StrOutputParser()

for concept in ["FAISS", "LangChain", "cosine similarity"]:
    result = chain.invoke({"concept": concept})
    print(f"  {concept:25} → {result}")

# ─────────────────────────────────────────────────────────────────
# 5. Sequential chain – two-step pipeline
# ─────────────────────────────────────────────────────────────────
print("\n[5] Sequential Chain – Define → Example")

define_prompt  = PromptTemplate.from_template("Define {term} in one sentence.")
example_prompt = PromptTemplate.from_template(
    "Given this definition: '{definition}'\nGive one real-world example."
)

define_chain  = define_prompt  | llm | StrOutputParser()
example_chain = example_prompt | llm | StrOutputParser()

term       = "transformer model"
definition = define_chain.invoke({"term": term})
example    = example_chain.invoke({"definition": definition})

print(f"\n  Term      : {term}")
print(f"  Definition: {definition}")
print(f"  Example   : {example}")

print("\n✅  Script 01 complete.")

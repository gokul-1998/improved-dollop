"""
Day 6 – Script 03: Tools & Agents (MCP Concepts)
=================================================
Demonstrates:
  - Creating custom tools with @tool decorator
  - ReAct agent with zero-shot reasoning
  - Integrating a free external API as a tool
  - MCP (Model Context Protocol) conceptual overview

Run:
    python 03_tools_agents.py

Requirements:
    pip install langchain langchain-community langchain-huggingface transformers accelerate requests
"""

import re
import requests
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline as hf_pipeline

print("=" * 60)
print("Day 6 – Script 03: Tools & Agents")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# 1. Define custom tools
# ─────────────────────────────────────────────────────────────────
print("\n[1] Defining Custom Tools")

@tool
def get_word_count(text: str) -> str:
    """Returns the number of words in the given text."""
    count = len(text.split())
    return f"The text contains {count} words."

@tool
def calculator(expression: str) -> str:
    """Safely evaluates a basic mathematical expression and returns the numeric result."""
    # Allow only safe characters
    if not re.match(r"^[0-9\s\+\-\*/\.\(\)]+$", expression):
        return "Error: Only basic arithmetic operators are allowed."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

@tool
def reverse_text(text: str) -> str:
    """Reverses the characters in the input text."""
    return text[::-1]

@tool
def get_joke(topic: str) -> str:
    """Fetches a random programming joke from the free JokeAPI. Topic is used as a search keyword."""
    try:
        url  = f"https://v2.jokeapi.dev/joke/Programming?type=single&contains={topic}"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if data.get("error"):
            return "Could not fetch a joke right now."
        return data.get("joke", "No joke found.")
    except Exception as e:
        return f"API unavailable: {e}"

# Test tools directly
print(f"  get_word_count: {get_word_count.invoke('LangChain makes AI easy to use')}")
print(f"  calculator    : {calculator.invoke('15 * 7 + 3')}")
print(f"  reverse_text  : {reverse_text.invoke('LangChain')}")

# ─────────────────────────────────────────────────────────────────
# 2. Load LLM
# ─────────────────────────────────────────────────────────────────
print("\n[2] Loading Flan-T5-base for agent reasoning...")
pipe = hf_pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=300)
llm  = HuggingFacePipeline(pipeline=pipe)
print("    Model loaded ✓")

# ─────────────────────────────────────────────────────────────────
# 3. ReAct Agent
# ─────────────────────────────────────────────────────────────────
print("\n[3] ReAct Agent – Reason + Act Loop")

tools = [get_word_count, calculator, reverse_text]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=4,
    handle_parsing_errors=True,
)

queries = [
    "How many words are in the sentence: 'Artificial intelligence is transforming the world'?",
    "What is the result of 144 divided by 12, then multiplied by 5?",
    "Reverse the word: 'Python'.",
]

for q in queries:
    print(f"\n  Query: {q}")
    try:
        answer = agent.run(q)
        print(f"  Final Answer: {answer}")
    except Exception as e:
        print(f"  Agent error: {e}")

# ─────────────────────────────────────────────────────────────────
# 4. MCP Conceptual Overview
# ─────────────────────────────────────────────────────────────────
print("\n[4] Model Context Protocol (MCP) – Conceptual Overview")
print("""
  MCP is an emerging standard for connecting LLMs to external tools.
  Think of it as "USB-C for AI agents":

  Traditional LangChain                 MCP
  ─────────────────────────────         ──────────────────────────────
  @tool Python decorator           →    JSON Schema manifest
  Hardcoded in agent               →    Server advertises capabilities
  In-process function call         →    HTTP / stdio transport
  Single-agent                     →    Multi-agent orchestration

  Example MCP tool manifest (JSON):
  {
    "name": "search_web",
    "description": "Search the internet for current information.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "Search query"}
      },
      "required": ["query"]
    }
  }
""")

# ─────────────────────────────────────────────────────────────────
# 5. External API tool demo
# ─────────────────────────────────────────────────────────────────
print("[5] External API Tool – Fetching a Joke")
joke = get_joke.invoke("python")
print(f"  Joke: {joke}")

print("\n✅  Script 03 complete.")

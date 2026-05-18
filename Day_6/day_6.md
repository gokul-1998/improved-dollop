# Day 6 – LangChain: Chains, Memory & Tools with Free LLMs

LangChain is an open-source framework for building applications powered by language models. Today you will master its three pillars — **Chains** (composable workflows), **Memory** (stateful conversations), and **Tools** (function-calling agents) — all using **free HuggingFace LLMs**, so no API keys or credit cards are needed.

---

## 1. LangChain Core Concepts

### What Is LangChain?

LangChain glues together the components you need to build LLM applications:

```
Prompt Template → LLM → Output Parser → Chain → Agent → Memory
```

| Component | Role |
|---|---|
| **PromptTemplate** | Parametric prompt with variables (`{topic}`, `{history}`) |
| **LLM / ChatModel** | The language model (local, HuggingFace, OpenAI, …) |
| **Chain** | Sequential or conditional composition of components |
| **Memory** | Persists conversation history between turns |
| **Tool** | A Python function the LLM can decide to call |
| **Agent** | An LLM that reasons and selects tools in a loop |

### Installation

```bash
pip install langchain langchain-community langchain-huggingface \
            transformers accelerate sentence-transformers
```

---

## 2. Part A — Prompt Templates & Chains

### 2.1 PromptTemplate Basics

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["topic", "level"],
    template="Explain {topic} to a {level} student in 3 bullet points.",
)

# Render the prompt
prompt_text = template.format(topic="transformers", level="beginner")
print(prompt_text)
```

### 2.2 ChatPromptTemplate (for chat models)

```python
from langchain.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI tutor specialising in {subject}."),
    ("human",  "{question}"),
])

messages = chat_prompt.format_messages(
    subject="machine learning",
    question="What is gradient descent?"
)
```

### 2.3 LLMChain – Prompt + Model + Output

```python
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# Free local LLM via HuggingFace
hf_pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=hf_pipe)

chain = LLMChain(llm=llm, prompt=template)
result = chain.run(topic="neural networks", level="high-school")
print(result)
```

### 2.4 Sequential Chains (pipe operator)

```python
# LangChain Expression Language (LCEL) — modern composition style
from langchain_core.output_parsers import StrOutputParser

chain = chat_prompt | llm | StrOutputParser()
response = chain.invoke({"subject": "Python", "question": "What is a decorator?"})
print(response)
```

### 2.5 Key Chain Types

| Chain | Purpose |
|---|---|
| `LLMChain` | Single prompt → LLM → output |
| `SequentialChain` | Output of one chain is input to the next |
| `TransformChain` | Apply a Python function in-chain |
| `RouterChain` | Route to different chains based on input |
| `RetrievalQA` | QA over documents using a retriever |

---

## 3. Part B — Conversation Memory

Memory lets a chain remember previous turns so the LLM can give contextually aware responses.

### 3.1 ConversationBufferMemory

Stores **all** messages verbatim — good for short conversations.

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()

conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

# Turn 1
r1 = conversation.predict(input="My name is Alice and I love NLP.")
print(r1)

# Turn 2 — the chain remembers the name
r2 = conversation.predict(input="What is my name?")
print(r2)

# Inspect stored history
print(memory.chat_memory.messages)
```

### 3.2 ConversationSummaryMemory

Summarises old turns to keep the context window small — ideal for long sessions.

```python
from langchain.memory import ConversationSummaryMemory

summary_memory = ConversationSummaryMemory(llm=llm)
conv_with_summary = ConversationChain(llm=llm, memory=summary_memory)
```

### 3.3 Memory Types Comparison

| Memory Type | Stores | Best For |
|---|---|---|
| `ConversationBufferMemory` | All messages (raw) | Short sessions |
| `ConversationBufferWindowMemory` | Last k messages | Medium sessions |
| `ConversationSummaryMemory` | LLM-generated summary | Long sessions |
| `ConversationTokenBufferMemory` | Last N tokens | Token-budget control |
| `VectorStoreRetrieverMemory` | Similar past messages | Large-scale retrieval |

---

## 4. Part C — Tools & Agents (MCP Concepts)

### 4.1 What Is a Tool?

A **Tool** is a Python function decorated so the LLM can discover and invoke it.

```python
from langchain.tools import tool

@tool
def get_word_count(text: str) -> int:
    """Returns the number of words in the given text."""
    return len(text.split())

@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression string and returns the result."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"
```

### 4.2 ReAct Agent — Reason + Act

The ReAct pattern lets the LLM alternate between reasoning (Thought) and action (Action) until it reaches an answer:

```
Thought: I need to count words in the summary.
Action: get_word_count
Action Input: "The model was trained on..."
Observation: 42
Thought: The word count is 42. I have the answer.
Final Answer: The summary contains 42 words.
```

```python
from langchain.agents import initialize_agent, AgentType

tools = [get_word_count, calculator]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

agent.run("How many words are in 'LangChain makes AI development easy'? Also, what is 15 * 7?")
```

### 4.3 Model Context Protocol (MCP) — Conceptual Overview

MCP is an emerging standard (pioneered by Anthropic) for connecting LLMs to external tools and APIs in a **structured, standardised way**:

| Concept | Traditional LangChain | MCP |
|---|---|---|
| Tool definition | Python `@tool` decorator | JSON Schema manifest |
| Discovery | Hardcoded in agent | Server advertises capabilities |
| Transport | In-process function call | HTTP / stdio protocol |
| Multi-agent | Manual wiring | Built-in orchestration |

MCP is analogous to **USB-C for AI tools** — any MCP-compliant server can plug into any MCP-compliant client.

```python
# Conceptual MCP tool manifest (JSON)
mcp_tool = {
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
```

### 4.4 Integrating External APIs as Tools

```python
import requests
from langchain.tools import tool

@tool
def get_joke(topic: str) -> str:
    """Fetches a random joke about the given topic from a free API."""
    try:
        resp = requests.get(
            f"https://v2.jokeapi.dev/joke/Any?contains={topic}&type=single",
            timeout=5
        )
        data = resp.json()
        return data.get("joke", "No joke found.")
    except Exception as e:
        return f"API error: {e}"
```

---

## 5. Hands-On Scripts

| File | What It Demonstrates |
|---|---|
| `01_chains_prompts.py` | PromptTemplate, LLMChain, LCEL pipe operator |
| `02_memory_conversation.py` | ConversationBufferMemory, SummaryMemory, multi-turn chat |
| `03_tools_agents.py` | Custom tools, ReAct agent, API integration, MCP concepts |

Run any script:
```bash
cd Day_6
python 01_chains_prompts.py
python 02_memory_conversation.py
python 03_tools_agents.py
```

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **PromptTemplate** | Parameterise prompts with variables; reuse across chains |
| **LLMChain / LCEL** | Compose prompt → LLM → parser in one line with `\|` |
| **Memory** | Buffer stores all history; Summary compresses it with an LLM |
| **Tool** | Any Python function annotated with `@tool` becomes callable by an agent |
| **ReAct Agent** | Reason-Act loop lets LLM decide *which* tool to call and *when* |
| **MCP** | Protocol standard for tool/API discovery — USB-C for AI agents |

---

## Further Reading

- [LangChain Documentation](https://python.langchain.com/docs/)
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/)
- [HuggingFace LangChain Integration](https://python.langchain.com/docs/integrations/llms/huggingface_hub)
- [Model Context Protocol (MCP) Spec](https://modelcontextprotocol.io/introduction)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)

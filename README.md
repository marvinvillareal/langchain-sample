# LangChain + LangGraph Tutorial (Ollama)

A practical, progressive tutorial for senior developers.  
All examples use **Ollama** for local LLM and embedding inference.

---

## Prerequisites

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull the models used in this tutorial
ollama pull llama3.2          # LLM
ollama pull nomic-embed-text  # Embeddings
```

### 3. Prepare documents (for RAG steps)
```bash
mkdir docs
echo "Authentication requires a valid API key passed in the Authorization header." > docs/auth.txt
echo "Rate limiting is applied per user at 100 requests per minute." > docs/rate_limits.txt
# Add any .txt files you want to query
```

---

## Files

| File | What it covers |
|------|----------------|
| `01_ingest.py` | Load, chunk, embed, and persist documents to Chroma |
| `02_retrieval.py` | Four retrieval strategies: basic, MMR, multi-query, compression |
| `03_chatbot.py` | Full LangGraph RAG chatbot with smart routing and memory |
| `04_chatbot_with_tools.py` | ReAct agent loop: LLM + tools (time, calculator, search stub) |
| `05_streaming.py` | Token streaming: direct LLM, LCEL chain, LangGraph |
| `06_structured_output.py` | Pydantic-validated LLM output: sentiment, NER, code review |
| `07_persistent_memory.py` | SQLite-backed memory that survives process restarts |

---

## Run Order

```bash
# 1. Ingest docs (required before 02, 03, 04)
python 01_ingest.py

# 2. Explore retrieval strategies
python 02_retrieval.py

# 3. Run the RAG chatbot
python 03_chatbot.py

# 4. Run the agent with tools
python 04_chatbot_with_tools.py

# 5. See streaming in action
python 05_streaming.py

# 6. See structured output
python 06_structured_output.py

# 7. Persistent memory chatbot
python 07_persistent_memory.py
```

---

## Architecture Overview

```
User Input
    │
    ▼
[LangGraph StateGraph]
    │
    ├── router node        → decides: RAG or direct?
    │       │
    │   needs_rag=True     needs_rag=False
    │       │                   │
    ▼       ▼                   ▼
[retrieve node]         [direct_generate node]
    │                           │
    ▼                           │
[rag_generate node]             │
    │                           │
    └─────────┬─────────────────┘
              ▼
           [END]
              │
    MemorySaver / SqliteSaver
    (persists state per thread_id)
```

---

## Swapping LLM Providers

All files are wired to Ollama but swapping is trivial:

```python
# Ollama (default)
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2")

# OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")

# Anthropic
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
```

Everything else stays the same — the Runnable interface is provider-agnostic.

---

## Next Steps

- **Add a real web search tool** in `04_chatbot_with_tools.py` using [Tavily](https://tavily.com)
- **Add a frontend** — stream tokens to a React/Next.js UI via Server-Sent Events
- **Multi-agent setup** — have multiple LangGraph subgraphs communicate via shared state
- **Evaluation** — use [LangSmith](https://smith.langchain.com) to trace, evaluate, and improve retrieval quality

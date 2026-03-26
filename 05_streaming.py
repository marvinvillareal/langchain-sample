"""
Step 5: Streaming Responses
----------------------------
Shows three streaming approaches:
  1. Direct LLM streaming (fastest, no graph overhead)
  2. LCEL chain streaming
  3. LangGraph stream_mode="messages" (token-level, full graph)

Usage:
    python 05_streaming.py
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

LLM_MODEL = "llama3.2"
QUESTION = "Explain how vector embeddings work in 3 sentences."


def demo_direct_stream():
    """Stream tokens directly from the LLM."""
    print("\n── 1. Direct LLM Streaming ─────────────────────────────────────")
    llm = ChatOllama(model=LLM_MODEL)
    print("Assistant: ", end="", flush=True)
    for chunk in llm.stream([HumanMessage(content=QUESTION)]):
        print(chunk.content, end="", flush=True)
    print()


def demo_chain_stream():
    """Stream tokens through an LCEL chain."""
    print("\n── 2. LCEL Chain Streaming ──────────────────────────────────────")
    llm = ChatOllama(model=LLM_MODEL)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise technical assistant."),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()

    print("Assistant: ", end="", flush=True)
    for token in chain.stream({"question": QUESTION}):
        print(token, end="", flush=True)
    print()


def demo_graph_stream():
    """Stream tokens from a LangGraph app (stream_mode='messages')."""
    print("\n── 3. LangGraph Message Streaming ───────────────────────────────")

    # Inline minimal graph so this file is self-contained
    from typing import Annotated
    from typing_extensions import TypedDict
    from langchain_core.messages import BaseMessage
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.checkpoint.memory import MemorySaver

    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    llm = ChatOllama(model=LLM_MODEL)

    def call_llm(state: State) -> State:
        system = SystemMessage(content="You are a concise technical assistant.")
        response = llm.invoke([system] + list(state["messages"]))
        return {"messages": [response]}

    graph = StateGraph(State)
    graph.add_node("llm", call_llm)
    graph.set_entry_point("llm")
    graph.add_edge("llm", END)
    app = graph.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "stream-demo"}}
    print("Assistant: ", end="", flush=True)

    for chunk, metadata in app.stream(
        {"messages": [HumanMessage(content=QUESTION)]},
        config=config,
        stream_mode="messages",
    ):
        # chunk is a message fragment; only print content from the llm node
        if metadata.get("langgraph_node") == "llm" and chunk.content:
            print(chunk.content, end="", flush=True)

    print()


if __name__ == "__main__":
    print(f"📡 Streaming demo — question: '{QUESTION}'")
    demo_direct_stream()
    demo_chain_stream()
    demo_graph_stream()
    print("\n✅ Done")

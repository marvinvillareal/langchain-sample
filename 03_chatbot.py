"""
Step 3: LangGraph RAG Chatbot
------------------------------
A multi-turn chatbot backed by:
  - Ollama LLM (llama3.2)
  - Chroma RAG (built in step 1)
  - LangGraph state machine with smart routing
  - MemorySaver checkpointer for conversation memory

Usage:
    python 03_chatbot.py
    python 03_chatbot.py --no-rag   # skip routing, always use direct LLM
"""

import argparse
from typing import Annotated

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"


# ── State ─────────────────────────────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # auto-appends per turn
    context: str    # retrieved document snippets
    needs_rag: bool # routing flag set by router node


# ── Component Factories ───────────────────────────────────────────────────────
def build_components():
    llm = ChatOllama(model=LLM_MODEL, temperature=0.7)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20},
    )
    return llm, retriever


# ── Nodes ─────────────────────────────────────────────────────────────────────
def make_router_node(llm):
    routing_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a routing assistant. Reply with ONLY 'yes' or 'no'.\n"
            "Does answering this question require looking up specific documents "
            "or knowledge base content?",
        ),
        ("human", "{question}"),
    ])
    chain = routing_prompt | llm | StrOutputParser()

    def router(state: ChatState) -> ChatState:
        question = state["messages"][-1].content
        result = chain.invoke({"question": question})
        needs = "yes" in result.lower()
        print(f"  [router] RAG needed: {needs}")
        return {"needs_rag": needs, "context": ""}

    return router


def make_retrieve_node(retriever):
    def retrieve(state: ChatState) -> ChatState:
        query = state["messages"][-1].content
        docs = retriever.invoke(query)
        context = "\n\n---\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            for doc in docs
        )
        print(f"  [retrieve] Found {len(docs)} chunk(s)")
        return {"context": context}

    return retrieve


def make_rag_generate_node(llm):
    def generate(state: ChatState) -> ChatState:
        system_content = (
            "You are a helpful assistant. Use ONLY the context below to answer "
            "the user's question. If the context doesn't contain the answer, "
            "say so clearly — do not make things up.\n\n"
            f"Context:\n{state['context']}"
        )
        messages = [SystemMessage(content=system_content)] + list(state["messages"])
        response = llm.invoke(messages)
        return {"messages": [response]}

    return generate


def make_direct_generate_node(llm):
    def generate(state: ChatState) -> ChatState:
        system = SystemMessage(content="You are a helpful assistant.")
        messages = [system] + list(state["messages"])
        response = llm.invoke(messages)
        return {"messages": [response]}

    return generate


# ── Routing Edge Logic ────────────────────────────────────────────────────────
def should_retrieve(state: ChatState) -> str:
    return "retrieve" if state.get("needs_rag", True) else "direct"


# ── Graph Builder ─────────────────────────────────────────────────────────────
def build_graph(use_rag: bool = True):
    llm, retriever = build_components()
    checkpointer = MemorySaver()

    graph = StateGraph(ChatState)

    # Register nodes
    graph.add_node("router", make_router_node(llm))
    graph.add_node("retrieve", make_retrieve_node(retriever))
    graph.add_node("rag_generate", make_rag_generate_node(llm))
    graph.add_node("direct_generate", make_direct_generate_node(llm))

    # Entry point
    graph.set_entry_point("router")

    # Conditional routing after router
    graph.add_conditional_edges(
        "router",
        should_retrieve,
        {"retrieve": "retrieve", "direct": "direct_generate"},
    )

    # Fixed edges
    graph.add_edge("retrieve", "rag_generate")
    graph.add_edge("rag_generate", END)
    graph.add_edge("direct_generate", END)

    return graph.compile(checkpointer=checkpointer)


# ── Chat Interface ────────────────────────────────────────────────────────────
def run_chat(use_rag: bool = True):
    print("🤖 LangGraph RAG Chatbot")
    print("   Model  :", LLM_MODEL)
    print("   RAG    :", "enabled" if use_rag else "disabled (--no-rag flag)")
    print("   Type 'exit' or 'quit' to stop, 'new' to start a fresh session.\n")

    app = build_graph(use_rag=use_rag)
    session_id = "session-1"
    session_count = 1

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if user_input.lower() == "new":
            session_count += 1
            session_id = f"session-{session_count}"
            print(f"  ✅ Started new session ({session_id})\n")
            continue

        config = {"configurable": {"thread_id": session_id}}
        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        answer = result["messages"][-1].content
        print(f"\nAssistant: {answer}\n")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph RAG chatbot")
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG routing (direct LLM only)",
    )
    args = parser.parse_args()
    run_chat(use_rag=not args.no_rag)

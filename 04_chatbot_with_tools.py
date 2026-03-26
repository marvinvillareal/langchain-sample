"""
Step 4: LangGraph Chatbot with Tools (ReAct Loop)
--------------------------------------------------
Extends the chatbot with tool-calling capability.
The LLM can decide to call tools mid-conversation and loop
until it has a final answer.

Included tools:
  - get_current_time   — returns the current date/time
  - calculate          — evaluates a math expression safely
  - web_search_mock    — stub you can replace with a real search API

Usage:
    python 04_chatbot_with_tools.py
"""

from datetime import datetime
from typing import Annotated

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"


# ── Tools ─────────────────────────────────────────────────────────────────────
@tool
def get_current_time() -> str:
    """Returns the current date and time."""
    return datetime.now().strftime("%A, %B %d %Y – %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a safe mathematical expression and return the result.
    Supports: +, -, *, /, **, parentheses, and basic math.
    Example: calculate("2 ** 10 + 3 * 7")
    """
    # Only allow safe characters to prevent code injection
    allowed = set("0123456789+-*/()., **eE")
    if not all(c in allowed for c in expression.replace(" ", "")):
        return "Error: expression contains disallowed characters."
    try:
        result = eval(expression, {"__builtins__": {}})  # no builtins
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def web_search_mock(query: str) -> str:
    """
    Search the web for current information.
    Replace this stub with a real API (e.g. Tavily, SerpAPI) in production.
    """
    # ↓ Replace with: from tavily import TavilyClient; client.search(query)
    return (
        f"[Mock search result for '{query}']\n"
        "This is a stub. Integrate a real search API here."
    )


TOOLS = [get_current_time, calculate, web_search_mock]


# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str


# ── Nodes ─────────────────────────────────────────────────────────────────────
def make_agent_node(llm_with_tools):
    """Main agent node — calls LLM with tools bound."""

    def agent(state: AgentState) -> AgentState:
        system = SystemMessage(
            content=(
                "You are a helpful assistant with access to tools. "
                "Use tools when they would help you give a better answer. "
                "If the user asks about documents or knowledge base content, "
                "mention that RAG retrieval is available separately.\n\n"
                + (
                    f"Document context (if retrieved):\n{state['context']}"
                    if state.get("context")
                    else ""
                )
            )
        )
        messages = [system] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return agent


def make_retrieve_node(retriever):
    """RAG retrieval node — called when agent decides context is needed."""

    def retrieve(state: AgentState) -> AgentState:
        query = state["messages"][-1].content
        docs = retriever.invoke(query)
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        print(f"  [retrieve] {len(docs)} chunk(s) loaded into context")
        return {"context": context}

    return retrieve


# ── Routing ───────────────────────────────────────────────────────────────────
def should_call_tools(state: AgentState) -> str:
    """If the last AI message has tool_calls, route to the tool node."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        tool_names = [tc["name"] for tc in last.tool_calls]
        print(f"  [router] calling tools: {tool_names}")
        return "tools"
    return "end"


# ── Graph Builder ─────────────────────────────────────────────────────────────
def build_agent():
    llm = ChatOllama(model=LLM_MODEL, temperature=0.5)
    llm_with_tools = llm.bind_tools(TOOLS)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", make_agent_node(llm_with_tools))
    graph.add_node("tools", tool_node)
    graph.add_node("retrieve", make_retrieve_node(retriever))

    # Entry → agent
    graph.set_entry_point("agent")

    # ReAct loop: agent → (tool call?) → tools → agent
    graph.add_conditional_edges(
        "agent",
        should_call_tools,
        {"tools": "tools", "end": END},
    )
    # After tools, always go back to agent so it can process the result
    graph.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ── Chat Interface ────────────────────────────────────────────────────────────
def run_chat():
    print("🤖 LangGraph Agent with Tools")
    print(f"   Model  : {LLM_MODEL}")
    print(f"   Tools  : {', '.join(t.name for t in TOOLS)}")
    print("   Type 'exit' to quit, 'new' for a new session.\n")

    app = build_agent()
    session_id = "agent-session-1"
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
            session_id = f"agent-session-{session_count}"
            print(f"  ✅ New session: {session_id}\n")
            continue

        config = {"configurable": {"thread_id": session_id}}

        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)], "context": ""},
            config=config,
        )
        answer = result["messages"][-1].content
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    run_chat()

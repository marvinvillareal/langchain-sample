"""
Step 7: Persistent Memory with SQLite
---------------------------------------
Uses SqliteSaver so conversation history survives process restarts.
Each unique thread_id is an independent conversation stored in a local DB.

Usage:
    python 07_persistent_memory.py              # interactive chat
    python 07_persistent_memory.py --list       # list all saved sessions
    python 07_persistent_memory.py --show <id>  # print history for a session
    python 07_persistent_memory.py --clear <id> # delete a session
"""

import argparse
import sqlite3
from typing import Annotated

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

DB_PATH = "./chat_history.db"
LLM_MODEL = "llama3.2"


# ── State ─────────────────────────────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── Graph ─────────────────────────────────────────────────────────────────────
def build_graph(checkpointer):
    llm = ChatOllama(model=LLM_MODEL, temperature=0.7)

    def call_llm(state: ChatState) -> ChatState:
        system = SystemMessage(
            content=(
                "You are a helpful assistant with persistent memory. "
                "You remember everything from earlier in this conversation."
            )
        )
        response = llm.invoke([system] + list(state["messages"]))
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("llm", call_llm)
    graph.set_entry_point("llm")
    graph.add_edge("llm", END)
    return graph.compile(checkpointer=checkpointer)


# ── Session Helpers ───────────────────────────────────────────────────────────
def list_sessions():
    """List all thread IDs stored in the SQLite checkpoint DB."""
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
        conn.close()
        if not rows:
            print("No saved sessions found.")
        else:
            print(f"Saved sessions ({len(rows)}):")
            for (tid,) in rows:
                print(f"  • {tid}")
    except sqlite3.OperationalError:
        print("No database found yet. Start a chat first.")


def show_session(thread_id: str, checkpointer: SqliteSaver):
    """Print the message history for a session."""
    config = {"configurable": {"thread_id": thread_id}}
    state = checkpointer.get(config)
    if state is None:
        print(f"Session '{thread_id}' not found.")
        return
    messages = state.get("channel_values", {}).get("messages", [])
    print(f"\nHistory for session '{thread_id}' ({len(messages)} messages):")
    for msg in messages:
        role = msg.__class__.__name__.replace("Message", "")
        print(f"  [{role}] {msg.content[:200]}")


# ── Interactive Chat ───────────────────────────────────────────────────────────
def run_chat(checkpointer: SqliteSaver):
    app = build_graph(checkpointer)

    print("🧠 Persistent Memory Chatbot")
    print(f"   Database : {DB_PATH}")
    print(f"   Model    : {LLM_MODEL}")
    print("   Commands : 'new' = new session, 'sessions' = list, 'exit' = quit\n")

    # Default session
    session_id = input("Session ID (press Enter for 'default'): ").strip() or "default"
    print(f"  ▶ Resuming or starting session: '{session_id}'\n")

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
            session_id = input("New session ID: ").strip() or f"session-{id(object())}"
            print(f"  ▶ Started session: '{session_id}'\n")
            continue

        if user_input.lower() == "sessions":
            list_sessions()
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List all sessions")
    parser.add_argument("--show", metavar="ID", help="Show history for a session")
    parser.add_argument("--clear", metavar="ID", help="Delete a session (not yet implemented)")
    args = parser.parse_args()

    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        if args.list:
            list_sessions()
        elif args.show:
            show_session(args.show, checkpointer)
        else:
            run_chat(checkpointer)

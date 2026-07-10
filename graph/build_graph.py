"""
Builds the LangGraph state machine: rewrite -> retrieve -> check -> generate/out_of_scope -> update_history.
Persists conversation state to a local SQLite file via LangGraph's checkpointer.
"""
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from graph.state import RAGState
from graph.nodes import (
    retrieve_node,
    check_relevance_node,
    generate_node,
    out_of_scope_node,
    route_after_relevance_check,
    rewrite_query_node,
    update_history_node,
)


def build_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("check_relevance", check_relevance_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("out_of_scope", out_of_scope_node)
    workflow.add_node("update_history", update_history_node)

    workflow.add_edge(START, "rewrite_query")
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "check_relevance")
    workflow.add_conditional_edges(
        "check_relevance",
        route_after_relevance_check,
        {"generate": "generate", "out_of_scope": "out_of_scope"},
    )
    workflow.add_edge("generate", "update_history")
    workflow.add_edge("out_of_scope", "update_history")
    workflow.add_edge("update_history", END)

    # Persistent checkpointing: raw sqlite3 connection, kept alive for the app's lifetime.
    # check_same_thread=False is required because LangGraph may access it from different threads.
    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return workflow.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    graph = build_graph()
    config = {"configurable": {"thread_id": "test-conversation-1"}}

    turns = [
        "What is the AI Risk Management Framework meant to help organizations do?",
        "What about its economic risks specifically?",
    ]

    for i, question in enumerate(turns):
        input_state = {
            "question": question,
            "rewritten_question": "",
            "retrieved_docs": [],
            "answer": "",
            "is_relevant": False,
        }
        if i == 0:
            input_state["chat_history"] = []  # only seed empty history on the very first turn

        result = graph.invoke(input_state, config=config)

        print(f"Question: {question}")
        print(f"Rewritten: {result['rewritten_question']}")
        print(f"Answer: {result['answer']}\n")
        print("-" * 60)
"""
Builds the LangGraph state machine with agentic routing + corrective retrieval (CRAG):
rewrite -> route -> [direct_answer | retrieve -> check_relevance -> grade_documents -> (generate -> citation | reformulate_query -> retrieve loop | out_of_scope) | decompose_retrieve -> generate -> citation] -> update_history
"""
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from graph.citation_node import citation_node
from graph.state import RAGState
from graph.nodes import (
    retrieve_node,
    check_relevance_node,
    generate_node,
    out_of_scope_node,
    route_after_relevance_check,
    rewrite_query_node,
    update_history_node,
    router_node,
    direct_answer_node,
    decompose_retrieve_node,
    route_after_classification,
    grade_documents_node,
    reformulate_query_node,
    route_after_grading,
)


def build_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("router", router_node)
    workflow.add_node("direct_answer", direct_answer_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("decompose_retrieve", decompose_retrieve_node)
    workflow.add_node("check_relevance", check_relevance_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("reformulate_query", reformulate_query_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("citation", citation_node)
    workflow.add_node("out_of_scope", out_of_scope_node)
    workflow.add_node("update_history", update_history_node)

    workflow.add_edge(START, "rewrite_query")
    workflow.add_edge("rewrite_query", "router")

    workflow.add_conditional_edges(
        "router",
        route_after_classification,
        {
            "direct_answer": "direct_answer",
            "retrieve": "retrieve",
            "decompose_retrieve": "decompose_retrieve",
            "out_of_scope": "out_of_scope",
        },
    )

    workflow.add_edge("retrieve", "check_relevance")
    workflow.add_conditional_edges(
        "check_relevance",
        route_after_relevance_check,
        {"generate": "grade_documents", "out_of_scope": "out_of_scope"},
    )

    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"generate": "generate", "reformulate": "reformulate_query", "out_of_scope": "out_of_scope"},
    )
    workflow.add_edge("reformulate_query", "retrieve")  # loop back to retry retrieval

    # Decomposed retrieval skips the relevance-check/grading gate and goes straight to generation,
    # since sub-questions were already router-approved as in-scope topics
    workflow.add_edge("decompose_retrieve", "generate")

    # generate always passes through citation validation/formatting before history update
    workflow.add_edge("generate", "citation")
    workflow.add_edge("citation", "update_history")

    # direct_answer and out_of_scope have no retrieved_docs to cite, so they
    # skip the citation node entirely and go straight to update_history
    workflow.add_edge("direct_answer", "update_history")
    workflow.add_edge("out_of_scope", "update_history")
    workflow.add_edge("update_history", END)

    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return workflow.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    graph = build_graph()

    test_questions = [
        "Hi, what can you help me with?",
        "What is the AI Risk Management Framework meant to help organizations do?",
        "How does climate change affect health, and what does WHO recommend for adaptation?",
        "What's the weather like tomorrow?",
    ]

    for i, question in enumerate(test_questions):
        config = {"configurable": {"thread_id": f"router-test-{i}"}}
        result = graph.invoke({
            "question": question,
            "rewritten_question": "",
            "retrieved_docs": [],
            "answer": "",
            "is_relevant": False,
            "chat_history": [],
            "route_category": "",
            "sub_questions": [],
            "retry_count": 0,
            "grading_passed": False,
        }, config=config)

        print(f"Question: {question}")
        print(f"Category: {result['route_category']}")
        if result["sub_questions"]:
            print(f"Sub-questions: {result['sub_questions']}")
        print(f"Retry count: {result['retry_count']}")
        print(f"Sources: {[d.metadata.get('source') for d in result['retrieved_docs']]}")
        print(f"Citations: {result.get('citations', [])}")
        print(f"Answer: {result['answer'][:400]}\n")
        print("-" * 60)

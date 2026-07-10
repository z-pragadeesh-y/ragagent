from langgraph.graph import StateGraph, START, END
from graph.state import RAGState
from graph.nodes import (
    retrieve_node,
    check_relevance_node,
    generate_node,
    out_of_scope_node,
    route_after_relevance_check,
    rewrite_query_node
)


def build_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("check_relevance", check_relevance_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("out_of_scope", out_of_scope_node)

    workflow.add_edge(START, "rewrite_query")
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "check_relevance")
    workflow.add_conditional_edges(
    "check_relevance",
    route_after_relevance_check,
    {"generate": "generate", "out_of_scope": "out_of_scope"},
)
    workflow.add_edge("generate", END)
    workflow.add_edge("out_of_scope", END)

    return workflow.compile()


if __name__ == "__main__":
    graph = build_graph()

    test_questions = [
        "What is the AI Risk Management Framework meant to help organizations do?",
        "What is the recipe for chocolate chip cookies?",
    ]

    for question in test_questions:
        result = graph.invoke({
            "question": question,
            "rewritten_question": "",
            "retrieved_docs": [],
            "answer": "",
            "is_relevant": False,
        })
        print(f"Question: {question}")
        print(f"Rewritten: {result['rewritten_question']}")
        print(f"Relevant: {result['is_relevant']}")
        print(f"Answer: {result['answer']}\n")
        print("-" * 60)
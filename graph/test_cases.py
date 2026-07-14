"""
Manual verification script: runs a batch of test questions across all 5 domains
plus edge cases, to sanity-check the full pipeline (Phases 1-4 + Plan 1 Steps 1-3)
before moving on. Each question runs in its own isolated thread_id, so results
never leak between unrelated test cases via shared conversation history.
"""
import uuid
from graph.build_graph import build_graph

TEST_CASES = [
    # (question, expected_relevant, expected_source_domain)
    ("What is the AI Risk Management Framework meant to help organizations do?", True, "document_1_AI"),
    ("What are the key findings of the IPCC AR6 report on climate change?", True, "document_2_Climate"),
    ("What does the IMF forecast for global economic growth?", True, "document_3_Economics"),
    ("What is the global life expectancy trend according to WHO?", True, "document_4_Health"),
    ("What is Retrieval-Augmented Generation and why is it used with LLMs?", True, "document_5_Natural"),

    # Edge cases
    ("What is the recipe for chocolate chip cookies?", False, None),
    ("Who won the FIFA World Cup in 2022?", False, None),
    ("What is the capital of France?", False, None),

    # Borderline / cross-domain (real stress test)
    ("How does climate change affect public health outcomes?", True, None),
    ("What economic risks does AI pose according to policy frameworks?", True, None),
]


def run_tests():
    graph = build_graph()
    results = []

    for question, expected_relevant, expected_domain in TEST_CASES:
        # Fresh, unique thread_id per test case -> no history leakage between test cases
        config = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}

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

        actual_relevant = result["is_relevant"]
        sources = [d.metadata.get("source", "") for d in result.get("retrieved_docs", [])]
        top_source = sources[0] if sources else "N/A"

        status = "✅" if actual_relevant == expected_relevant else "❌"

        results.append({
            "question": question,
            "expected_relevant": expected_relevant,
            "actual_relevant": actual_relevant,
            "top_source": top_source,
            "answer_preview": result["answer"][:150],
            "status": status,
        })

    return results


if __name__ == "__main__":
    results = run_tests()
    for r in results:
        print(f"{r['status']} Q: {r['question']}")
        print(f"   Expected relevant: {r['expected_relevant']} | Actual: {r['actual_relevant']}")
        print(f"   Top source: {r['top_source']}")
        print(f"   Answer: {r['answer_preview']}...")
        print()

    passed = sum(1 for r in results if r["status"] == "✅")
    print(f"\n{passed}/{len(results)} passed")

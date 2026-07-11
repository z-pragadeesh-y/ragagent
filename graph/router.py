"""
Agentic router: classifies a query into one of 4 paths before retrieval.
"""
import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

ROUTER_PROMPT_TEMPLATE = """You are a query router for a RAG system with a knowledge base covering exactly
these 5 topics: AI policy/risk management (NIST AI RMF), climate change (IPCC AR6), global economics (IMF
World Economic Outlook), public health (WHO statistics), and AI/NLP research (Retrieval-Augmented Generation).

Classify the user's question into EXACTLY ONE of these categories:
- "direct": greetings, meta questions about the assistant itself, or anything not requiring the knowledge base
- "simple": a single, focused question answerable by retrieving from ONE topic area
- "decompose": a question that genuinely combines TWO OR MORE of the 5 topics above and needs separate retrieval for each part
- "out_of_scope": a question clearly unrelated to any of the 5 topics

Respond with ONLY a JSON object, no other text, in this exact format:
{{"category": "simple", "sub_questions": []}}

For "decompose", split the question into 2-3 standalone sub-questions in "sub_questions".
For all other categories, "sub_questions" should be an empty list.

Question: {question}

JSON response:"""


def route_query(question: str) -> dict:
    """Returns {'category': ..., 'sub_questions': [...]}."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT_TEMPLATE)
    chain = prompt | llm

    response = chain.invoke({"question": question})
    raw = response.content.strip()

    # Strip markdown code fences if the model adds them despite instructions
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()

    try:
        parsed = json.loads(raw)
        if parsed.get("category") not in {"direct", "simple", "decompose", "out_of_scope"}:
            raise ValueError("invalid category")
        return parsed
    except (json.JSONDecodeError, ValueError):
        # Safe fallback: if the router output is malformed, treat as "simple"
        # rather than crashing the whole graph.
        return {"category": "simple", "sub_questions": []}


if __name__ == "__main__":
    test_questions = [
        "Hi, what can you help me with?",
        "What is the AI Risk Management Framework meant to help organizations do?",
        "How does climate change affect health, and what does WHO recommend for adaptation?",
        "What's the weather like tomorrow?",
    ]

    for q in test_questions:
        result = route_query(q)
        print(f"Q: {q}")
        print(f"   Category: {result['category']}")
        if result["sub_questions"]:
            print(f"   Sub-questions: {result['sub_questions']}")
        print()
"""
Agentic router: classifies a query into one of 4 paths before retrieval.
Uses the SIMPLE lane (NVIDIA NIM -> local LM Studio) since routing is a
structured, low-reasoning, JSON-shaped task - Groq's quota is reserved for
the complex/generation lane.

FIX (decompose + upload gap): route_query now accepts has_uploaded_doc.
Previously the router only knew about the fixed 5-domain KB, so a question
mixing an uploaded document with one of the 5 domains (e.g. "how does the
UDHR's approach to human dignity compare to WHO life expectancy stats")
could never be classified as "decompose" - the uploaded-doc half of the
question isn't one of the router's known topics, so the whole question
either got misread as "simple" (losing half the question) or "out_of_scope"
(losing all of it). The prompt now explicitly tells the router when this
thread has an uploaded document, and adds it as a valid decompose target.
"""
import json
import logging
from langchain_core.prompts import ChatPromptTemplate

from llm.task_router import get_llm
from llm.errors import AllProvidersFailedError

logger = logging.getLogger("llm_manager")

ROUTER_PROMPT_TEMPLATE = """You are a query router for a RAG system with a knowledge base covering exactly
these 5 topics: AI policy/risk management (NIST AI RMF), climate change (IPCC AR6), global economics (IMF
World Economic Outlook), public health (WHO statistics), and AI/NLP research (Retrieval-Augmented Generation).

{uploaded_doc_context}

Classify the user's question into EXACTLY ONE of these categories:
- "direct": greetings, meta questions about the assistant itself, or anything not requiring the knowledge base
- "simple": a single, focused question answerable by retrieving from ONE topic area (this includes a
  question answerable entirely from the uploaded document alone, if one is attached)
- "decompose": a question that genuinely combines TWO OR MORE distinct sources - this includes combining
  TWO of the 5 topics above, OR combining the uploaded document (if attached) with ONE of the 5 topics above
- "out_of_scope": a question clearly unrelated to any of the 5 topics AND unrelated to the uploaded document
  (if one is attached)

Respond with ONLY a JSON object, no other text, in this exact format:
{{"category": "simple", "sub_questions": []}}

For "decompose", split the question into 2-3 standalone sub-questions in "sub_questions". If one part of
the question is about the uploaded document, phrase that sub-question so it stands alone and clearly
refers to the uploaded document's subject matter.
For all other categories, "sub_questions" should be an empty list.

Question: {question}

JSON response:"""

NO_UPLOAD_CONTEXT = "This conversation currently has no uploaded document attached."
HAS_UPLOAD_CONTEXT = (
    "This conversation currently has a document attached (uploaded by the user this session). "
    "Treat it as a valid additional source alongside the 5 fixed topics above - a question can "
    "combine it with any of the 5 topics and should be classified as \"decompose\" in that case."
)


def route_query(question: str, has_uploaded_doc: bool = False) -> dict:
    """Returns {'category': ..., 'sub_questions': [...]}.
    Falls back to 'out_of_scope' (the safe default) if every provider fails,
    or if the LLM's output can't be parsed as valid routing JSON.

    has_uploaded_doc: whether the calling thread currently has a session
    document attached (graph/nodes.py's router_node passes this in from
    state["has_uploaded_doc"], which main.py seeds via session_store.has_session_doc).
    Lets the router treat the upload as a valid decompose target instead of
    only knowing about the fixed 5-domain KB."""
    llm = get_llm(task="route", temperature=0)
    prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT_TEMPLATE)
    chain = prompt | llm

    uploaded_doc_context = HAS_UPLOAD_CONTEXT if has_uploaded_doc else NO_UPLOAD_CONTEXT

    try:
        response = chain.invoke({"question": question, "uploaded_doc_context": uploaded_doc_context})
        raw = response.content.strip()
    except AllProvidersFailedError:
        logger.error("Router: all providers failed - defaulting to simple for safety")
        return {"category": "simple", "sub_questions": []}

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
        # rather than crashing the whole graph. This is a SEPARATE safety net
        # from the provider-failover one above - it handles the case where a
        # provider DID respond, but its response wasn't valid routing JSON.
        logger.warning(f"Router: could not parse LLM output as valid JSON: {raw!r} - defaulting to simple")
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

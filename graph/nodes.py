"""
Graph nodes: each takes RAGState, returns a partial state update.
Every LLM call goes through get_llm(task=...) (llm/task_router.py), which
picks the right lane:
  - SIMPLE lane (NVIDIA NIM -> local LM Studio): rewrite, route, grade
  - COMPLEX lane (Groq -> NVIDIA NIM -> local LM Studio): generate, decompose
Nodes never know or care which provider actually answered - they only handle
the case where an entire lane failed (AllProvidersFailedError), degrading
gracefully instead of crashing.
"""
import logging
from langchain_core.prompts import ChatPromptTemplate

from ingestion.vectorstore import load_vectorstore
from ingestion.hybrid_retriever import hybrid_retrieve
from graph.state import RAGState
from graph.router import route_query
from llm.task_router import get_llm
from llm.errors import AllProvidersFailedError

logger = logging.getLogger("llm_manager")

RETRIEVAL_K = 4  # hybrid retrieval + reranking is precise enough; no longer need k=8 as a band-aid
MAX_RETRIES = 2  # CRAG: max reformulate-and-retry attempts before giving up

ALL_PROVIDERS_DOWN_MESSAGE = "All configured LLM providers are currently unavailable. Please try again shortly."

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the provided context.
Synthesize an answer from the relevant parts of the context, even if the information is spread across
multiple passages or is only partially related. Only say "I don't have enough information to answer that"
if the context truly contains nothing related to the question.

Context:
{context}

Question: {question}

Answer:"""

REWRITE_PROMPT_TEMPLATE = """Given the conversation history and a new question, rewrite the new question
to be a clear, standalone, specific question optimized for semantic search — resolving any pronouns or
implied references (like "it", "that", "its") using the conversation history.

IMPORTANT: Preserve any acronyms, technical terms, or proper nouns EXACTLY as the user wrote them
(e.g., "RAG", "HyDE", "NIST", "IPCC", "IMF", "WHO") — do NOT expand or replace them with their full form,
since exact terminology matters for search accuracy.

If the question is already standalone and specific, return it unchanged. Do not answer the question,
only rewrite it. Return ONLY the rewritten question, nothing else.

Conversation history:
{history}

New question: {question}

Rewritten question:"""

GRADE_PROMPT_TEMPLATE = """You are grading whether retrieved passages are relevant to a question.
For each numbered passage below, respond with ONLY "yes" or "no" on its own line, in order.
Do not add any other text.

Question: {question}

{passages}

Respond with one yes/no per line, {count} lines total:"""


def retrieve_node(state: RAGState) -> dict:
    """Retrieves relevant chunks using hybrid (BM25 + vector) search with cross-encoder reranking.
    Uses HyDE (hypothetical document embeddings) for the vector search leg."""
    docs = hybrid_retrieve(state["rewritten_question"], fusion_k=15, final_k=RETRIEVAL_K, use_hyde=True)
    return {"retrieved_docs": docs}


def check_relevance_node(state: RAGState) -> dict:
    """Checks relevance based on whether hybrid retrieval actually found any candidates."""
    is_relevant = len(state.get("retrieved_docs", [])) > 0
    return {"is_relevant": is_relevant}


def generate_node(state: RAGState) -> dict:
    """Generates an answer from the retrieved chunks. COMPLEX lane: Groq -> NVIDIA -> local."""
    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source')}]\n{doc.page_content}"
        for doc in state["retrieved_docs"]
    )

    llm = get_llm(task="generate", temperature=0)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm

    try:
        response = chain.invoke({"context": context, "question": state["question"]})
        answer = response.content
    except AllProvidersFailedError:
        answer = ALL_PROVIDERS_DOWN_MESSAGE

    return {"answer": answer}


def out_of_scope_node(state: RAGState) -> dict:
    """Returns a fixed response when no relevant content was found."""
    return {"answer": "I don't have information about that in my knowledge base."}


def route_after_relevance_check(state: RAGState) -> str:
    """Conditional edge function: decides which node runs next."""
    return "generate" if state["is_relevant"] else "out_of_scope"


def rewrite_query_node(state: RAGState) -> dict:
    """Rewrites the raw question into a clearer, standalone, retrieval-friendly form.
    SIMPLE lane: NVIDIA -> local (Groq excluded to protect its quota)."""
    history_text = "\n".join(
        f"{turn['role']}: {turn['content']}" for turn in state.get("chat_history", [])
    ) or "(no previous turns)"

    llm = get_llm(task="rewrite", temperature=0)
    prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT_TEMPLATE)
    chain = prompt | llm

    try:
        response = chain.invoke({"history": history_text, "question": state["question"]})
        rewritten = response.content.strip()
    except AllProvidersFailedError:
        # Fall back to the raw, unrewritten question rather than crashing
        rewritten = state["question"]

    return {"rewritten_question": rewritten}


def update_history_node(state: RAGState) -> dict:
    """Appends the current Q&A turn to chat history, for use in future turns."""
    history = state.get("chat_history", [])
    updated = history + [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": state["answer"]},
    ]
    return {"chat_history": updated}


def router_node(state: RAGState) -> dict:
    """Classifies the rewritten question into a routing category."""
    result = route_query(state["rewritten_question"])
    return {"route_category": result["category"], "sub_questions": result["sub_questions"]}


def direct_answer_node(state: RAGState) -> dict:
    """Handles greetings/meta questions without any retrieval. COMPLEX lane, since
    it's user-facing conversational output where quality matters most."""
    llm = get_llm(task="generate", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant for a knowledge base covering AI policy, climate change, "
        "economics, public health, and AI research. Respond briefly and naturally to: {question}"
    )
    chain = prompt | llm

    try:
        response = chain.invoke({"question": state["question"]})
        answer = response.content
    except AllProvidersFailedError:
        answer = ALL_PROVIDERS_DOWN_MESSAGE

    return {"answer": answer}


def decompose_retrieve_node(state: RAGState) -> dict:
    """Retrieves separately for each sub-question, then combines results.
    (Pure retrieval - no LLM call here; the synthesis happens afterward in
    generate_node, which is why 'decompose' is a COMPLEX-lane task name even
    though this specific function doesn't call an LLM itself.)"""
    all_docs = []
    seen_content = set()
    for sub_q in state["sub_questions"]:
        docs = hybrid_retrieve(sub_q, fusion_k=15, final_k=3)
        for doc in docs:
            if doc.page_content not in seen_content:
                all_docs.append(doc)
                seen_content.add(doc.page_content)
    return {"retrieved_docs": all_docs, "is_relevant": len(all_docs) > 0}


def route_after_classification(state: RAGState) -> str:
    """Conditional edge: decides graph path based on router category."""
    category = state["route_category"]
    if category == "direct":
        return "direct_answer"
    elif category == "decompose":
        return "decompose_retrieve"
    elif category == "out_of_scope":
        return "out_of_scope"
    else:  # "simple"
        return "retrieve"


def grade_documents_node(state: RAGState) -> dict:
    """Grades all retrieved docs in a SINGLE LLM call. SIMPLE lane: NVIDIA -> local."""
    docs = state["retrieved_docs"]
    if not docs:
        return {"retrieved_docs": [], "grading_passed": False}

    passages_text = "\n\n".join(
        f"Passage {i+1}: {doc.page_content[:800]}" for i, doc in enumerate(docs)
    )

    llm = get_llm(task="grade", temperature=0)
    prompt = ChatPromptTemplate.from_template(GRADE_PROMPT_TEMPLATE)
    chain = prompt | llm

    try:
        response = chain.invoke({
            "question": state["rewritten_question"],
            "passages": passages_text,
            "count": len(docs),
        })
    except AllProvidersFailedError:
        # If we can't grade at all, trust retrieval as-is rather than crashing
        return {"retrieved_docs": docs, "grading_passed": True}

    verdicts = [line.strip().lower() for line in response.content.strip().split("\n") if line.strip()]

    graded_docs = [
        doc for doc, verdict in zip(docs, verdicts)
        if "yes" in verdict
    ]

    passed = len(graded_docs) >= 1  # even one genuinely relevant chunk is enough to attempt an answer
    return {"retrieved_docs": graded_docs, "grading_passed": passed}


def reformulate_query_node(state: RAGState) -> dict:
    """Rewrites the query differently after a failed grading pass, and increments retry count.
    SIMPLE lane, same reasoning as rewrite_query_node."""
    llm = get_llm(task="rewrite", temperature=0.3)  # slight variation so retries don't repeat the same phrasing
    prompt = ChatPromptTemplate.from_template(
        "The following search query did not retrieve good results from a knowledge base covering "
        "AI policy, climate change, economics, public health, and AI research. Rewrite it using "
        "different phrasing or more specific/alternative terms, while preserving any acronyms and "
        "technical terms exactly as written. Return ONLY the rewritten query.\n\n"
        "Original query: {question}\n\nRewritten query:"
    )
    chain = prompt | llm

    try:
        response = chain.invoke({"question": state["rewritten_question"]})
        new_query = response.content.strip()
    except AllProvidersFailedError:
        # Can't reformulate right now - keep the same query but still count the
        # attempt, so we don't loop forever waiting on providers to recover
        new_query = state["rewritten_question"]

    return {
        "rewritten_question": new_query,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def route_after_grading(state: RAGState) -> str:
    """Conditional edge: retry retrieval, give up, or proceed to generation."""
    if state["grading_passed"]:
        return "generate"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "out_of_scope"
    return "reformulate"

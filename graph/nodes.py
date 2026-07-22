"""
Graph nodes: each takes RAGState, returns a partial state update.
Every LLM call goes through get_llm(task=...) (llm/task_router.py), which
picks the right lane:
  - SIMPLE lane (NVIDIA NIM -> local LM Studio): rewrite, route, grade, hyde
  - COMPLEX lane (Groq -> NVIDIA NIM -> local LM Studio): generate, decompose
Nodes never know or care which provider actually answered - they only handle
the case where an entire lane failed (AllProvidersFailedError), degrading
gracefully instead of crashing.

Plan 2 Step 3: generate_node labels each retrieved chunk [Source N] in the
context and instructs the model to cite inline. graph/citation_node.py (run
right after this node) validates those citations against the real retrieved
chunks and appends the actual References list - generate_node itself does
NOT build references, to avoid duplicating that responsibility.

Plan 3: retrieve_node now also merges in session-scoped uploaded-document
chunks for this thread_id (ingestion/session_store.py), if any exist.
Uploaded chunks are tagged domain_tag="uploaded" and are completely
isolated from the permanent KB - they only ever surface for the thread_id
that uploaded them. route_after_classification also gets a bypass: if the
router says out_of_scope but this thread has an uploaded doc, we still
attempt retrieval, since the router only knows the fixed 5-domain KB and
has no visibility into session uploads.
"""
import logging
from langchain_core.prompts import ChatPromptTemplate

from ingestion.vectorstore import load_vectorstore
from ingestion.hybrid_retriever import hybrid_retrieve
from ingestion.session_store import has_session_doc, session_retrieve
from graph.state import RAGState
from graph.router import route_query
from llm.task_router import get_llm
from llm.errors import AllProvidersFailedError

logger = logging.getLogger("llm_manager")

RETRIEVAL_K = 4
MAX_RETRIES = 2

ALL_PROVIDERS_DOWN_MESSAGE = "All configured LLM providers are currently unavailable. Please try again shortly."

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the provided context.
Synthesize an answer from the relevant parts of the context, even if the information is spread across
multiple passages or is only partially related. Only say "I don't have enough information to answer that"
if the context truly contains nothing related to the question.

Each passage below is labeled [Source N]. When you use information from a passage, cite it inline with
its label, e.g. "AI risks include bias and security concerns [Source 1]." Cite every factual claim.

Context:
{context}

Question: {question}

Answer (with inline [Source N] citations):"""

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


def _build_labeled_context(docs) -> str:
    """Labels each chunk [Source N] in the context fed to the LLM, so it can cite
    inline. Does NOT build the References list - that's citation_node's job,
    using real metadata, so the LLM's citation numbers can be validated against it."""
    return "\n\n---\n\n".join(
        f"[Source {i}]\n{doc.page_content}" for i, doc in enumerate(docs, start=1)
    )


def retrieve_node(state: RAGState) -> dict:
    """Retrieves relevant chunks using hybrid (BM25 + vector) search with cross-encoder
    reranking. Uses HyDE for the vector search leg. Plan 3: if this thread_id has a
    session-scoped uploaded document, its chunks are retrieved separately and placed
    FIRST (uploaded docs are usually what the user is directly asking about), then
    merged with the permanent-KB results."""
    docs = hybrid_retrieve(state["rewritten_question"], fusion_k=15, final_k=RETRIEVAL_K, use_hyde=True)

    thread_id = state.get("thread_id", "")
    if thread_id and has_session_doc(thread_id):
        session_docs = session_retrieve(thread_id, state["rewritten_question"], k=RETRIEVAL_K)
        docs = session_docs + docs

    return {"retrieved_docs": docs}


def check_relevance_node(state: RAGState) -> dict:
    """Checks relevance based on whether hybrid retrieval actually found any candidates."""
    is_relevant = len(state.get("retrieved_docs", [])) > 0
    return {"is_relevant": is_relevant}


def generate_node(state: RAGState) -> dict:
    """Generates an answer with inline [Source N] citations. COMPLEX lane."""
    docs = state["retrieved_docs"]
    context = _build_labeled_context(docs)

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
    """Handles greetings/meta questions without any retrieval. COMPLEX lane."""
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
    """Retrieves separately for each sub-question, then combines results."""
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
    """Conditional edge: decides graph path based on router category.
    Plan 3: if the router says out_of_scope but this thread has an uploaded
    document, we still attempt retrieval - the router only knows the fixed
    5-domain KB, so out_of_scope here means "not one of the 5 domains", not
    "definitely unanswerable"."""
    category = state["route_category"]
    if category == "direct":
        return "direct_answer"
    elif category == "decompose":
        return "decompose_retrieve"
    elif category == "out_of_scope":
        if state.get("has_uploaded_doc"):
            return "retrieve"
        return "out_of_scope"
    else:
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
        return {"retrieved_docs": docs, "grading_passed": True}

    verdicts = [line.strip().lower() for line in response.content.strip().split("\n") if line.strip()]

    graded_docs = [
        doc for doc, verdict in zip(docs, verdicts)
        if "yes" in verdict
    ]

    passed = len(graded_docs) >= 1
    return {"retrieved_docs": graded_docs, "grading_passed": passed}


def reformulate_query_node(state: RAGState) -> dict:
    """Rewrites the query differently after a failed grading pass, and increments retry count."""
    llm = get_llm(task="rewrite", temperature=0.3)
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

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

Priority 1 (Dynamic Document Ingestion) rewrite: retrieve_node and
decompose_retrieve_node both look up this thread's session index (if any)
via ingestion/session_store.py and pass it into hybrid_retrieve(), which
fuses session chunks into the SAME RRF+rerank ranking as the permanent
corpus - there is no separate retrieval path for uploads, and no
concatenation after the fact. This also closes the previous gap where
decompose_retrieve_node never checked for uploads at all: both entry
points now share one retrieval call shape.

Priority 1 hardening (post-verification fix):
1. route_after_classification no longer hard-stops at out_of_scope_node when
   the router returns "out_of_scope" for a KB-domain question it just failed
   to topic-match (e.g. "rare earth magnets" not obviously matching
   "economics" to the router LLM). Retrieval + grade_documents_node (CRAG)
   are a far stronger, evidence-based signal than one router LLM guess, so
   retrieval now always runs and grading is the real gate. A genuinely
   out-of-scope question will still correctly fail grading after retries and
   land on out_of_scope_node - this just removes a false negative from a
   single-point-of-failure classification step.
2. RETRIEVAL_K raised 4 -> 6 and PROMPT_TEMPLATE tightened so multi-part
   "compare X vs Y vs Z" questions (e.g. Govern/Map/Measure) get enough
   chunks and the model states the answer directly when the context
   supports it, instead of hedging ("not explicitly stated") when the
   information is actually present but split across passages.
"""
import logging
from langchain_core.prompts import ChatPromptTemplate

from ingestion.hybrid_retriever import hybrid_retrieve
from ingestion.session_store import has_session_doc, get_session_vectorstore, get_session_bm25
from graph.state import RAGState
from graph.router import route_query
from llm.task_router import get_llm
from llm.errors import AllProvidersFailedError

logger = logging.getLogger("llm_manager")

RETRIEVAL_K = 6  # was 4 - gives multi-part/compare questions enough chunk coverage
MAX_RETRIES = 2

ALL_PROVIDERS_DOWN_MESSAGE = "All configured LLM providers are currently unavailable. Please try again shortly."

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the provided context.
Synthesize an answer from the relevant parts of the context, even if the information is spread across
multiple passages or is only partially related. Only say "I don't have enough information to answer that"
if the context truly contains nothing related to the question.

If the context contains the answer, even partially or across multiple passages, state it directly and
confidently - do not say "not explicitly stated" or hedge if the relevant information is present. Synthesize
across passages rather than declaring the answer missing.

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


def _get_session_retrieval_kwargs(thread_id: str) -> dict:
    """Looks up this thread's session index (uploaded doc), if any, and returns
    the kwargs hybrid_retrieve() needs to fuse it into the unified ranking.
    Returns an empty dict if the thread has no upload - hybrid_retrieve()
    behaves identically to the permanent-corpus-only case in that scenario."""
    if not thread_id or not has_session_doc(thread_id):
        return {}
    session_vectorstore = get_session_vectorstore(thread_id)
    session_bm25, session_chunks = get_session_bm25(thread_id)
    return {
        "session_vectorstore": session_vectorstore,
        "session_bm25": session_bm25,
        "session_chunks": session_chunks,
    }


def retrieve_node(state: RAGState) -> dict:
    """Retrieves relevant chunks using hybrid (BM25 + vector) search with cross-encoder
    reranking. Uses HyDE for the vector search leg. If this thread has an uploaded
    document, its chunks are fused into the SAME ranking via hybrid_retrieve()'s
    session_* kwargs - not retrieved separately, not concatenated afterward."""
    session_kwargs = _get_session_retrieval_kwargs(state.get("thread_id", ""))
    docs = hybrid_retrieve(
        state["rewritten_question"], fusion_k=15, final_k=RETRIEVAL_K, use_hyde=True, **session_kwargs
    )
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


NO_INFO_PHRASE = "i don't have enough information to answer that"


def route_after_generate(state: RAGState) -> str:
    """
    Priority 4 safety net: grade_documents_node's single-call LLM grading can
    occasionally false-positive ("yes, relevant") on chunks that are actually
    unrelated (e.g. climate-report chunks passing grading for a weather
    question) - when that happens, generate_node correctly and honestly
    produces its instructed refusal phrase (PROMPT_TEMPLATE's "I don't have
    enough information to answer that"), but without this check that dead-end
    answer would just be returned directly, since grading technically
    "passed" and never routes through out_of_scope/web_search_node at all.

    This catches that exact refusal phrase and redirects to out_of_scope
    (graph/web_search_node.py) instead - giving the question a real shot at
    a web-search answer rather than silently defeating Priority 4's purpose
    whenever the grader has a false positive. Does NOT touch grading logic
    itself; this is a downstream safety net, not a grading fix.
    """
    answer = state.get("answer", "").strip().lower()
    if NO_INFO_PHRASE in answer:
        return "out_of_scope"
    return "citation"


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
    """Classifies the rewritten question into a routing category.
    FIX: now passes has_uploaded_doc through to route_query, so the router
    can classify a question mixing the uploaded doc + a permanent-KB domain
    as "decompose" instead of only recognizing the fixed 5 topics."""
    result = route_query(state["rewritten_question"], has_uploaded_doc=state.get("has_uploaded_doc", False))
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
    """Retrieves separately for each sub-question, then combines results.
    Priority 1 fix: now also fuses this thread's session doc (if any) into
    EVERY sub-question's retrieval, via the same hybrid_retrieve() call
    shape as retrieve_node - previously this path never checked for uploads
    at all."""
    session_kwargs = _get_session_retrieval_kwargs(state.get("thread_id", ""))
    all_docs = []
    seen_content = set()
    for sub_q in state["sub_questions"]:
        docs = hybrid_retrieve(sub_q, fusion_k=15, final_k=3, **session_kwargs)
        for doc in docs:
            if doc.page_content not in seen_content:
                all_docs.append(doc)
                seen_content.add(doc.page_content)
    return {"retrieved_docs": all_docs, "is_relevant": len(all_docs) > 0}


def route_after_classification(state: RAGState) -> str:
    """Conditional edge: decides graph path based on router category.

    FIXED: previously, an "out_of_scope" verdict only proceeded to retrieval
    if the thread happened to have an uploaded doc - otherwise it hard-stopped
    at out_of_scope_node, meaning a single router LLM misclassification (e.g.
    a niche in-KB subtopic like rare-earth magnets not obviously reading as
    "economics" to the router) could permanently block a real, in-KB answer
    with no recourse.

    Now, retrieval always runs regardless of category, and check_relevance_node
    + grade_documents_node (CRAG) - which check the actual retrieved evidence,
    not a topic guess - are the real gate for whether the question is
    answerable. A genuinely out-of-scope question still correctly lands on
    out_of_scope_node after failing grading through MAX_RETRIES reformulation
    attempts (see route_after_grading below). This trades a small amount of
    wasted retrieval compute on true out-of-scope questions for eliminating
    false negatives on in-KB questions the router mis-tagged.
    """
    category = state["route_category"]
    if category == "direct":
        return "direct_answer"
    elif category == "decompose":
        return "decompose_retrieve"
    else:
        # "simple" and "out_of_scope" both go through retrieval now -
        # grading is the real arbiter, not the router's topic guess.
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

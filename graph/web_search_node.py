"""
Priority 4: web search fallback via Tavily, replacing the old fixed
out_of_scope refusal message.

Reached via the SAME "out_of_scope" edges route_after_relevance_check and
route_after_grading already produce - CRAG's grading is still the real gate
deciding a question couldn't be answered from the KB (see nodes.py's
route_after_classification/route_after_grading docstrings). This node just
answers a different way once that gate is hit, instead of refusing outright.
build_graph.py wires this node in as the "out_of_scope" target - no changes
needed to either routing function's return values.

Web results are wrapped as Document objects with the SAME metadata shape
citation_node.py already expects (source, document_title, section_title,
domain_tag) - domain_tag="web" distinguishes them from KB chunks without
requiring ANY special-casing in citation_node itself. The LLM cites them as
[Source N] exactly like any other retrieved chunk, and citation_node builds
the References list from real Tavily metadata (title, url) - never invented.

scope_guard_node's prompt was updated (not bypassed) to treat an answer
explicitly labeled as web-sourced as in-scope by construction, while still
catching genuine opinion/advice drift on top of it - see scope_guard_node.py.

Falls back to a fixed refusal message (no retrieved_docs, so citation_node
and scope_guard both pass through cleanly with nothing to check) if
TAVILY_API_KEY is unset, the search call fails, or returns zero results -
never fabricates an answer from nothing.
"""
import logging
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from graph.state import RAGState
from llm.config import get_settings
from llm.task_router import get_llm
from llm.errors import AllProvidersFailedError

logger = logging.getLogger("llm_manager")

KB_ONLY_FALLBACK_MESSAGE = "I don't have information about that in my knowledge base."
NO_WEB_RESULTS_MESSAGE = (
    "I don't have information about that in my knowledge base, and couldn't "
    "find anything relevant on the web either."
)

WEB_ANSWER_PROMPT = """You are answering a question using web search results, since it falls outside your
knowledge base (AI policy, climate, economics, public health, AI research). Synthesize an answer from the
passages below, citing inline with [Source N] for every factual claim - same citation style you'd use for
any other context.

IMPORTANT: Begin your answer with a brief note that this is from a web search, not the knowledge base -
e.g. "Based on a web search: ..." - so the user always knows this isn't grounded in the built-in KB.

Context:
{context}

Question: {question}

Answer (with inline [Source N] citations):"""


def _get_tavily_client():
    """Returns a TavilyClient, or None if TAVILY_API_KEY isn't configured -
    callers treat None the same as a search that found nothing."""
    settings = get_settings()
    if not settings.tavily_api_key:
        return None
    from tavily import TavilyClient
    return TavilyClient(api_key=settings.tavily_api_key)


def _search_web(query: str, max_results: int = 5) -> list[Document]:
    """Runs a Tavily search and wraps each result as a Document with the
    SAME metadata shape citation_node.py already reads for KB chunks -
    domain_tag="web" is the only signal distinguishing these from KB
    results, and citation_node needs no awareness of that distinction at all."""
    client = _get_tavily_client()
    if client is None:
        return []

    results = []
    try:
        response = client.search(query=query, max_results=max_results, search_depth="advanced")
        results = response.get("results", [])
    except Exception as exc:
        logger.warning("web_search_node: Tavily search (advanced) failed (%s), trying basic", exc)

    if not results:
        try:
            response = client.search(query=query, max_results=max_results, search_depth="basic")
            results = response.get("results", [])
        except Exception as exc:
            logger.warning("web_search_node: Tavily search (basic) failed (%s)", exc)
            return []

    docs = []
    for result in results:
        content = result.get("content", "")
        if not content:
            continue
        docs.append(Document(
            page_content=content,
            metadata={
                "source": result.get("url", "unknown"),
                "document_title": result.get("title", "Web result"),
                "section_title": "",
                "domain_tag": "web",
            },
        ))
    return docs


def web_search_node(state: RAGState) -> dict:
    """Reached when the KB (retrieve/decompose_retrieve + CRAG grading via
    grade_documents_node) couldn't answer the question. Tries a real Tavily
    web search before falling back to a fixed refusal message - never
    fabricates an answer without real search results backing it. Sets
    retrieved_docs so citation_node (which runs right after this node, same
    as the generate path) can attach real References from the web results.

    By this point the question may have gone through 1-2 rounds of
    reformulate_query_node, which deliberately biases rewrites toward the
    KB's own domain phrasing (climate/economics/health/etc terms) to help
    KB retrieval - that bias actively hurts a general web search, so this
    node uses the ORIGINAL state["question"] instead of rewritten_question,
    unlike retrieve_node/decompose_retrieve_node which want the opposite."""
    question = state["question"]
    docs = _search_web(question)

    if not docs:
        settings = get_settings()
        message = NO_WEB_RESULTS_MESSAGE if settings.tavily_api_key else KB_ONLY_FALLBACK_MESSAGE
        return {"answer": message, "retrieved_docs": []}

    context = "\n\n---\n\n".join(
        f"[Source {i}]\n{doc.page_content}" for i, doc in enumerate(docs, start=1)
    )

    llm = get_llm(task="generate", temperature=0)
    prompt = ChatPromptTemplate.from_template(WEB_ANSWER_PROMPT)
    chain = prompt | llm

    try:
        response = chain.invoke({"context": context, "question": question})
        answer = response.content
    except AllProvidersFailedError:
        return {"answer": KB_ONLY_FALLBACK_MESSAGE, "retrieved_docs": []}

    return {"answer": answer, "retrieved_docs": docs}

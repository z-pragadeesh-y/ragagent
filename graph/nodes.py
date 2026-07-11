"""
Graph nodes: each takes RAGState, returns a partial state update.
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from ingestion.vectorstore import load_vectorstore
from graph.state import RAGState
from ingestion.hybrid_retriever import hybrid_retrieve
load_dotenv()

RETRIEVAL_K = 4  # hybrid retrieval + reranking is more precise, so we no longer need k=8 as a band-aid  # increased from 4 to reduce missed-context failures on broad questions

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the provided context.
Synthesize an answer from the relevant parts of the context, even if the information is spread across
multiple passages or is only partially related. Only say "I don't have enough information to answer that"
if the context truly contains nothing related to the question.

Context:
{context}

Question: {question}

Answer:"""


def retrieve_node(state: RAGState) -> dict:
    """Retrieves relevant chunks using hybrid (BM25 + vector) search with cross-encoder reranking."""
    docs = hybrid_retrieve(state["rewritten_question"], fusion_k=15, final_k=RETRIEVAL_K)
    return {"retrieved_docs": docs}


def check_relevance_node(state: RAGState) -> dict:
    """Checks relevance based on whether hybrid retrieval actually found any candidates."""
    is_relevant = len(state.get("retrieved_docs", [])) > 0
    return {"is_relevant": is_relevant}


def generate_node(state: RAGState) -> dict:
    """Generates an answer from the retrieved chunks using Groq."""
    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source')}]\n{doc.page_content}"
        for doc in state["retrieved_docs"]
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm

    response = chain.invoke({"context": context, "question": state["question"]})
    return {"answer": response.content}


def out_of_scope_node(state: RAGState) -> dict:
    """Returns a fixed response when no relevant content was found."""
    return {"answer": "I don't have information about that in my knowledge base."}


def route_after_relevance_check(state: RAGState) -> str:
    """Conditional edge function: decides which node runs next."""
    return "generate" if state["is_relevant"] else "out_of_scope"



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


def rewrite_query_node(state: RAGState) -> dict:
    """Rewrites the raw question into a clearer, standalone, retrieval-friendly form."""
    history_text = "\n".join(
        f"{turn['role']}: {turn['content']}" for turn in state.get("chat_history", [])
    ) or "(no previous turns)"

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT_TEMPLATE)
    chain = prompt | llm

    response = chain.invoke({"history": history_text, "question": state["question"]})
    rewritten = response.content.strip()

    return {"rewritten_question": rewritten}
def update_history_node(state: RAGState) -> dict:
    """Appends the current Q&A turn to chat history, for use in future turns."""
    history = state.get("chat_history", [])
    updated = history + [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": state["answer"]},
    ]
    return {"chat_history": updated} 
from graph.router import route_query
from ingestion.hybrid_retriever import hybrid_retrieve

def router_node(state: RAGState) -> dict:
    """Classifies the rewritten question into a routing category."""
    result = route_query(state["rewritten_question"])
    return {"route_category": result["category"], "sub_questions": result["sub_questions"]}


def direct_answer_node(state: RAGState) -> dict:
    """Handles greetings/meta questions without any retrieval."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant for a knowledge base covering AI policy, climate change, "
        "economics, public health, and AI research. Respond briefly and naturally to: {question}"
    )
    chain = prompt | llm
    response = chain.invoke({"question": state["question"]})
    return {"answer": response.content}


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
   
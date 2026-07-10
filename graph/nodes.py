"""
Graph nodes: each takes RAGState, returns a partial state update.
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from ingestion.vectorstore import load_vectorstore
from graph.state import RAGState

load_dotenv()

RETRIEVAL_K = 8  # increased from 4 to reduce missed-context failures on broad questions

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the provided context.
Synthesize an answer from the relevant parts of the context, even if the information is spread across
multiple passages or is only partially related. Only say "I don't have enough information to answer that"
if the context truly contains nothing related to the question.

Context:
{context}

Question: {question}

Answer:"""


def retrieve_node(state: RAGState) -> dict:
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(state["rewritten_question"], k=RETRIEVAL_K)
    return {"retrieved_docs": docs}


def check_relevance_node(state: RAGState) -> dict:
    vectorstore = load_vectorstore()
    results_with_scores = vectorstore.similarity_search_with_score(
        state["rewritten_question"], k=RETRIEVAL_K
    )
    is_relevant = any(distance < 1.0 for _, distance in results_with_scores)
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
implied references (like "it", "that", "its") using the conversation history. If the question is already
standalone and specific, return it unchanged. Do not answer the question, only rewrite it.
Return ONLY the rewritten question, nothing else.

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
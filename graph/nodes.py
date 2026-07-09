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
    """Retrieves top-k relevant chunks for the question."""
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(state["question"], k=RETRIEVAL_K)
    return {"retrieved_docs": docs}


def check_relevance_node(state: RAGState) -> dict:
    """Checks if retrieved docs are similar enough to bother generating an answer."""
    vectorstore = load_vectorstore()
    results_with_scores = vectorstore.similarity_search_with_score(
        state["question"], k=RETRIEVAL_K
    )

    # Chroma's default here returns L2 distance: LOWER = more similar (not a 0-1 score)
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
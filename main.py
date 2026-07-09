"""
Phase 1: Vanilla RAG — retrieve relevant chunks, generate an answer with Groq.
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from ingestion.vectorstore import load_vectorstore, build_vectorstore
from pathlib import Path

load_dotenv()

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""


def get_vectorstore():
    """Loads existing vector store, or builds it if it doesn't exist yet."""
    chroma_path = Path(__file__).resolve().parent / ".chroma"
    if chroma_path.exists():
        return load_vectorstore()
    return build_vectorstore()


def answer_question(question: str, k: int = 4):
    vectorstore = get_vectorstore()
    retrieved_docs = vectorstore.similarity_search(question, k=k)

    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source')}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm

    response = chain.invoke({"context": context, "question": question})

    return response.content, retrieved_docs


if __name__ == "__main__":
    question = "What are the main challenges in AI risk management according to NIST?"

    print(f"Question: {question}\n")
    answer, sources = answer_question(question)

    print(f"Answer:\n{answer}\n")
    print("--- Sources used ---")
    for doc in sources:
        print(f"- {doc.metadata.get('source')}")
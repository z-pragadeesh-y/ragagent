"""
Shared state object that flows through the LangGraph pipeline.
"""
from typing import TypedDict, List
from langchain_core.documents import Document


class RAGState(TypedDict):
    question: str
    rewritten_question: str
    retrieved_docs: List[Document]
    answer: str
    is_relevant: bool
    chat_history: List[dict]  # [{"role": "user"/"assistant", "content": "..."}]
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
    chat_history: List[dict]
    route_category: str
    sub_questions: List[str]
    retry_count: int
    grading_passed: bool
    citations: List[dict]
    is_injection: bool

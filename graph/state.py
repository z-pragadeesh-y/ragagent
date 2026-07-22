"""
Shared state object that flows through the LangGraph pipeline.

Plan 3 (doc upload): added thread_id and has_uploaded_doc. thread_id lets
retrieve_node look up whether THIS conversation has a session-scoped
uploaded document to merge into retrieval (ingestion/session_store.py).
has_uploaded_doc is set once per invocation (from main.py, which already
knows this cheaply via session_store.has_session_doc) so route_after_
classification can bypass an out_of_scope router verdict without every
node needing to re-check disk/cache itself.
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
    scope_flagged: bool
    thread_id: str
    has_uploaded_doc: bool

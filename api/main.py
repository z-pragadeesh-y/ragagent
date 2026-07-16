"""
FastAPI layer: /chat endpoint with SSE streaming and semantic caching.
Fully independent of the graph's internal nodes - can be run standalone.

NOTE ON STREAMING: the graph is compiled with a synchronous SqliteSaver
checkpointer (see graph/build_graph.py), which does not support LangGraph's
async astream(). Rather than rewrite the checkpointer to an async variant
(a change that would ripple into every caller of build_graph() - test_cases.py,
custom_ragas.py, the __main__ block), /chat/stream runs the existing synchronous
graph.invoke() inside a thread pool (via run_in_threadpool) so it doesn't block
the server's event loop, then sends the complete answer as a single SSE event.
This is honest streaming infrastructure (async-safe, non-blocking, SSE-formatted)
but does NOT stream individual tokens as they're generated - true token-level
streaming would require the async checkpointer migration described above,
noted here as a known follow-up rather than done now.
"""
import json
import uuid
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from graph.build_graph import build_graph
from api.semantic_cache import get_cache

app = FastAPI(title="RAGAgent API")
graph = build_graph()
cache = get_cache()


class ChatRequest(BaseModel):
    question: str
    thread_id: str = None


def _build_initial_state(question: str, is_new_thread: bool) -> dict:
    """
    Builds the initial graph state for one invocation. chat_history is only
    seeded with an empty list on a genuinely NEW thread - if thread_id was
    already provided by the caller, we must NOT include chat_history at all,
    so the checkpointer's restored history from prior turns isn't silently
    overwritten (this exact bug was found and fixed back in Phase 4 - see
    LEARNING_NOTES.md - and reappeared here in the new API layer until caught).
    """
    state = {
        "question": question,
        "rewritten_question": "",
        "retrieved_docs": [],
        "answer": "",
        "is_relevant": False,
        "route_category": "",
        "sub_questions": [],
        "retry_count": 0,
        "grading_passed": False,
    }
    if is_new_thread:
        state["chat_history"] = []
    return state


def _run_graph_sync(question: str, thread_id: str, is_new_thread: bool) -> str:
    """Runs the full graph synchronously. Called via run_in_threadpool so it
    never blocks the async event loop."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(_build_initial_state(question, is_new_thread), config=config)
    return result["answer"]


@app.post("/chat")
async def chat(request: ChatRequest):
    """Non-streaming endpoint with semantic caching - returns the full answer at once."""
    cached_answer = cache.get(request.question)
    if cached_answer is not None:
        return {"answer": cached_answer, "from_cache": True, "thread_id": request.thread_id}

    is_new_thread = request.thread_id is None
    thread_id = request.thread_id or str(uuid.uuid4())

    answer = await run_in_threadpool(_run_graph_sync, request.question, thread_id, is_new_thread)

    cache.set(request.question, answer)
    return {"answer": answer, "from_cache": False, "thread_id": thread_id}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    SSE endpoint. On a cache hit, returns instantly as one event. On a cache
    miss, runs the graph in a thread pool (non-blocking) and sends the
    complete answer as a single SSE event once ready - see module docstring
    for why this isn't true token-by-token streaming yet.
    """
    cached_answer = cache.get(request.question)
    is_new_thread = request.thread_id is None
    thread_id = request.thread_id or str(uuid.uuid4())

    async def event_generator():
        if cached_answer is not None:
            yield f"data: {json.dumps({'token': cached_answer, 'from_cache': True, 'thread_id': thread_id})}\n\n"
            yield "data: [DONE]\n\n"
            return

        answer = await run_in_threadpool(_run_graph_sync, request.question, thread_id, is_new_thread)
        cache.set(request.question, answer)

        yield f"data: {json.dumps({'token': answer, 'from_cache': False, 'thread_id': thread_id})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}

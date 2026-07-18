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

Plan 2 Step 3: _run_graph_sync returns {"answer", "citations"} instead of a
bare string, since graph/citation_node.py attaches structured citations
(source, section, domain_tag) on state alongside the answer text. Both
endpoints and the semantic cache store/return that full dict.

Plan 2 Step 5: after every graph invocation (cache hits are intentionally
NOT logged again - only genuine new invocations), a feedback row is fired
off to feedback/feedback_logger.py in a background thread, so logging can
never add latency to or block the actual response. Wrapped so a logging
failure is fully invisible to the request path.
"""
import json
import threading
import uuid
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from graph.build_graph import build_graph
from api.semantic_cache import get_cache
from feedback.feedback_logger import log_feedback

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
        "citations": [],
    }
    if is_new_thread:
        state["chat_history"] = []
    return state


def _fire_feedback_log(thread_id: str, question: str, result_state: dict) -> None:
    """Spawns a background thread to log this turn - never awaited, never
    blocks the response path. Any exception inside log_feedback itself is
    already swallowed internally, so this is just about not even waiting
    on the write to complete before responding."""
    retrieved_sources = [
        d.metadata.get("source") for d in result_state.get("retrieved_docs", [])
    ]
    threading.Thread(
        target=log_feedback,
        kwargs={
            "thread_id": thread_id,
            "question": question,
            "answer": result_state.get("answer", ""),
            "citations": result_state.get("citations", []),
            "retrieved_sources": retrieved_sources,
            "route_category": result_state.get("route_category", ""),
            "is_injection": result_state.get("is_injection", False),
            "scope_flagged": result_state.get("scope_flagged", False),
        },
        daemon=True,
    ).start()


def _run_graph_sync(question: str, thread_id: str, is_new_thread: bool) -> dict:
    """Runs the full graph synchronously. Called via run_in_threadpool so it
    never blocks the async event loop. Returns both the answer text and the
    structured citations list built by citation_node. Also fires off a
    feedback log row for this turn before returning."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(_build_initial_state(question, is_new_thread), config=config)

    _fire_feedback_log(thread_id, question, result)

    return {"answer": result["answer"], "citations": result.get("citations", [])}


@app.post("/chat")
async def chat(request: ChatRequest):
    """Non-streaming endpoint with semantic caching - returns the full answer at once."""
    cached = cache.get(request.question)
    if cached is not None:
        return {
            "answer": cached["answer"],
            "citations": cached.get("citations", []),
            "from_cache": True,
            "thread_id": request.thread_id,
        }

    is_new_thread = request.thread_id is None
    thread_id = request.thread_id or str(uuid.uuid4())

    result = await run_in_threadpool(_run_graph_sync, request.question, thread_id, is_new_thread)

    cache.set(request.question, result)
    return {
        "answer": result["answer"],
        "citations": result.get("citations", []),
        "from_cache": False,
        "thread_id": thread_id,
    }


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    SSE endpoint. On a cache hit, returns instantly as one event. On a cache
    miss, runs the graph in a thread pool (non-blocking) and sends the
    complete answer as a single SSE event once ready - see module docstring
    for why this isn't true token-by-token streaming yet.
    """
    cached = cache.get(request.question)
    is_new_thread = request.thread_id is None
    thread_id = request.thread_id or str(uuid.uuid4())

    async def event_generator():
        if cached is not None:
            yield f"data: {json.dumps({'token': cached['answer'], 'citations': cached.get('citations', []), 'from_cache': True, 'thread_id': thread_id})}\n\n"
            yield "data: [DONE]\n\n"
            return

        result = await run_in_threadpool(_run_graph_sync, request.question, thread_id, is_new_thread)
        cache.set(request.question, result)

        yield f"data: {json.dumps({'token': result['answer'], 'citations': result.get('citations', []), 'from_cache': False, 'thread_id': thread_id})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}

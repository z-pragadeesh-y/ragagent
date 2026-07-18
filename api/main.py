"""
FastAPI layer: /chat endpoint with SSE streaming and semantic caching.
Fully independent of the graph's internal nodes - can be run standalone.

NOTE ON STREAMING: see graph/build_graph.py docstring - /chat/stream runs the
synchronous graph.invoke() inside a thread pool rather than true token-level
async streaming (documented tradeoff, not a bug).

Plan 2 Step 3 fix: _run_graph_sync now returns a dict ({"answer", "citations"})
instead of a plain string, since generate_node's answer now includes citations
built by citation_node. The semantic cache stores/returns that whole dict
consistently, instead of mixing dict and string types across code paths.
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
    overwritten (this exact bug was found and fixed back in Phase 4).
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


def _run_graph_sync(question: str, thread_id: str, is_new_thread: bool) -> dict:
    """Runs the full graph synchronously. Called via run_in_threadpool so it
    never blocks the async event loop. Returns {'answer': str, 'citations': list}."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(_build_initial_state(question, is_new_thread), config=config)
    return {"answer": result["answer"], "citations": result.get("citations", [])}


@app.post("/chat")
async def chat(request: ChatRequest):
    """Non-streaming endpoint with semantic caching - returns the full answer at once."""
    cached_result = cache.get(request.question)
    if cached_result is not None:
        return {
            "answer": cached_result["answer"],
            "citations": cached_result.get("citations", []),
            "from_cache": True,
            "thread_id": request.thread_id,
        }

    is_new_thread = request.thread_id is None
    thread_id = request.thread_id or str(uuid.uuid4())

    result = await run_in_threadpool(_run_graph_sync, request.question, thread_id, is_new_thread)

    cache.set(request.question, result)
    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "from_cache": False,
        "thread_id": thread_id,
    }


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    SSE endpoint. On a cache hit, returns instantly as one event. On a cache
    miss, runs the graph in a thread pool (non-blocking) and sends the
    complete answer as a single SSE event once ready.
    """
    cached_result = cache.get(request.question)
    is_new_thread = request.thread_id is None
    thread_id = request.thread_id or str(uuid.uuid4())

    async def event_generator():
        if cached_result is not None:
            payload = {
                "token": cached_result["answer"],
                "citations": cached_result.get("citations", []),
                "from_cache": True,
                "thread_id": thread_id,
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
            return

        result = await run_in_threadpool(_run_graph_sync, request.question, thread_id, is_new_thread)
        cache.set(request.question, result)

        payload = {
            "token": result["answer"],
            "citations": result["citations"],
            "from_cache": False,
            "thread_id": thread_id,
        }
        yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}

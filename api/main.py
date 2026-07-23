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

FRONTEND PREP: _run_graph_sync now also returns route_category, is_injection,
and scope_flagged - these already existed on RAGState but were never
surfaced past the graph boundary. They power the frontend's "Pipeline
Inspector" panel. Cache hits replay the ORIGINAL result dict (already
containing these fields from when first computed), so cached responses
stay consistent with fresh ones instead of returning blank pipeline data.

Plan 3: POST /upload and DELETE /upload/{thread_id} let a user attach a
personal document (.pdf/.txt/.md) to a specific conversation thread. The
doc is embedded into a session-scoped Chroma collection keyed by thread_id
(ingestion/session_store.py) - NEVER merged into the permanent 5-domain KB.
_build_initial_state now also seeds thread_id and has_uploaded_doc onto
graph state, since graph/nodes.py's retrieve_node needs thread_id to look
up whether this conversation has a session doc to merge into retrieval.
"""
import json
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from graph.build_graph import build_graph
from api.semantic_cache import get_cache
from feedback.feedback_logger import log_feedback
from ingestion.session_store import (
    build_session_vectorstore,
    has_session_doc,
    delete_session_vectorstore,
)

app = FastAPI(title="RAGAgent API")

# CORS: the frontend runs on a different origin than this API (localhost:5173
# during development, a Vercel domain once deployed), so the browser blocks
# fetch() calls unless this API explicitly allows it. allow_origins is kept
# as an explicit list rather than "*" so credentials/cookies could be added
# later without hitting the browser's "wildcard + credentials" restriction.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        # TODO: add the real Vercel production URL here once deployed,
        # e.g. "https://ragagent-frontend.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()
cache = get_cache()

ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB - generous for text/policy-doc-sized PDFs


class ChatRequest(BaseModel):
    question: str
    thread_id: str = None


def _build_initial_state(question: str, thread_id: str, is_new_thread: bool) -> dict:
    """
    Builds the initial graph state for one invocation. chat_history is only
    seeded with an empty list on a genuinely NEW thread - if thread_id was
    already provided by the caller, we must NOT include chat_history at all,
    so the checkpointer's restored history from prior turns isn't silently
    overwritten (this exact bug was found and fixed back in Phase 4 - see
    LEARNING_NOTES.md - and reappeared here in the new API layer until caught).

    Plan 3: thread_id and has_uploaded_doc are now seeded here too, so
    retrieve_node (graph/nodes.py) can look up and merge in this thread's
    session-scoped uploaded document, if one exists.
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
        "thread_id": thread_id,
        "has_uploaded_doc": has_session_doc(thread_id),
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
    never blocks the async event loop. Returns the answer text, structured
    citations, and pipeline metadata (route_category, is_injection,
    scope_flagged) for the frontend's Pipeline Inspector. Also fires off a
    feedback log row for this turn before returning."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(_build_initial_state(question, thread_id, is_new_thread), config=config)

    _fire_feedback_log(thread_id, question, result)

    return {
        "answer": result["answer"],
        "citations": result.get("citations", []),
        "route_category": result.get("route_category", ""),
        "is_injection": result.get("is_injection", False),
        "scope_flagged": result.get("scope_flagged", False),
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    """Non-streaming endpoint with semantic caching - returns the full answer at once."""
    cached = cache.get(request.question)
    if cached is not None:
        return {
            "answer": cached["answer"],
            "citations": cached.get("citations", []),
            "route_category": cached.get("route_category", ""),
            "is_injection": cached.get("is_injection", False),
            "scope_flagged": cached.get("scope_flagged", False),
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
        "route_category": result.get("route_category", ""),
        "is_injection": result.get("is_injection", False),
        "scope_flagged": result.get("scope_flagged", False),
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
            payload = {
                "token": cached["answer"],
                "citations": cached.get("citations", []),
                "route_category": cached.get("route_category", ""),
                "is_injection": cached.get("is_injection", False),
                "scope_flagged": cached.get("scope_flagged", False),
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
            "citations": result.get("citations", []),
            "route_category": result.get("route_category", ""),
            "is_injection": result.get("is_injection", False),
            "scope_flagged": result.get("scope_flagged", False),
            "from_cache": False,
            "thread_id": thread_id,
        }
        yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    thread_id: str = Form(None),
):
    """
    Plan 3: accepts a .pdf/.txt/.md file, embeds it into a session-scoped
    Chroma collection keyed by thread_id (never touching the permanent KB),
    and returns the thread_id to use for subsequent /chat calls on this
    conversation. If thread_id isn't provided, a new one is generated - the
    frontend should always send the returned thread_id back on /chat so the
    uploaded doc is actually visible to that conversation's retrieval.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_UPLOAD_EXTENSIONS)}",
        )

    thread_id = thread_id or str(uuid.uuid4())

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail=f"File too large - max {MAX_UPLOAD_BYTES // (1024 * 1024)}MB",
            )
        tmp.write(contents)

    try:
        chunk_count = await run_in_threadpool(
            build_session_vectorstore, thread_id, tmp_path, file.filename
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "thread_id": thread_id,
        "filename": file.filename,
        "chunks_stored": chunk_count,
        "status": "ingested",
    }


@app.delete("/upload/{thread_id}")
async def delete_upload(thread_id: str):
    """Removes a thread's uploaded document. Call this when a conversation
    ends, or when the user explicitly wants to remove their uploaded doc
    without ending the whole conversation."""
    await run_in_threadpool(delete_session_vectorstore, thread_id)
    return {"thread_id": thread_id, "status": "deleted"}


@app.get("/health")
async def health():
    return {"status": "ok"}

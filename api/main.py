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

FIX (large-upload 504 timeout): /upload was previously synchronous - the
whole request (save file -> extract text -> markdown_generator's sequential
windowed LLM calls -> chunk -> embed) had to complete inside ONE HTTP
request, so any document needing enough windows to exceed Cloud Run's
request timeout (900s) failed with a 504, no matter how high that timeout
was set - the real bottleneck is unbounded, timeout-ceiling-agnostic work
happening synchronously inside a request/response cycle.

/upload now returns almost immediately with {"status": "processing"} and
kicks off ingestion in a background thread. A new in-memory status table
(_upload_status) tracks each thread_id's progress; GET /upload/status/
{thread_id} lets the frontend poll until status flips to "ready" or "error".
This is intentionally a simple in-process dict, not a database/queue - safe
because Cloud Run is running with --max-instances=1 (single container, see
Cloud Run migration notes), so there's no multi-instance consistency
problem. If max-instances is ever raised above 1, this table needs to move
to a shared store (Redis, Firestore, etc.) since each instance would only
see its own in-memory copy.

FRONTEND NOTE: this is a backend-only fix. The frontend's upload flow must
be updated to (1) read {"status": "processing"} from the initial POST
/upload response instead of expecting {"status": "ingested"} immediately,
and (2) poll GET /upload/status/{thread_id} (e.g. every 2-3s) until it sees
{"status": "ready"} or {"status": "error"}, before allowing the user to ask
questions against the upload. Until the frontend is updated to poll, uploads
will appear to "hang" in the UI even though they're now succeeding server-side.

Priority 2 (real per-node Pipeline Inspector data): /chat/stream previously
sent ONE SSE event containing the complete answer, with the frontend faking
node-by-node progress on a fixed setTimeout schedule unrelated to what the
graph actually did. Misleading on every branch the timer didn't know about
(direct_answer skips retrieval entirely, out_of_scope stops early, a failed
grading pass loops back through reformulate_query -> retrieve again).

Key fact this relies on: the checkpointer is synchronous (SqliteSaver), but
graph.stream(..., stream_mode="updates") is ALSO synchronous and still
yields one {node_name: state_update} dict per node as it completes - no
async checkpointer migration required. _stream_graph_worker runs this
generator in a background thread and pushes each node event onto a
thread-safe queue.Queue; the async SSE generator drains it via
loop.run_in_executor(None, q.get), which blocks the executor thread (not
the event loop) between events. Real incremental streaming, not simulated
timing.

_serialize_node_update() strips each node's raw state update to a small,
JSON-safe, frontend-relevant slice (doc counts/sources, route category,
retry count, flags) - never the full retrieved chunk text or in-progress
answer.

Cache hits skip the graph entirely, so they can't emit real per-node
events - they send a single "cache_hit" event instead of faking a node
sequence.
"""
import asyncio
import json
import queue
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from graph.build_graph import build_graph
from graph.nodes import ALL_PROVIDERS_DOWN_MESSAGE
from graph.web_search_node import KB_ONLY_FALLBACK_MESSAGE, NO_WEB_RESULTS_MESSAGE
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
        "https://ragagent-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()
cache = get_cache()

ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB - generous for text/policy-doc-sized PDFs

# In-memory upload status table: thread_id -> {"status": ..., "filename": ...,
# "chunks_stored": ..., "error": ...}. See module docstring FIX note above
# for why this is safe only at --max-instances=1.
_upload_status: dict = {}
_upload_status_lock = threading.Lock()


class ChatRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None


# Priority 4 bug fix: these are all TRANSIENT failure fallbacks (an LLM
# provider outage, or - as actually happened during Priority 4 testing - a
# local DNS resolution failure making Tavily unreachable), never a genuine,
# stable answer to the question. Caching one of these means every future
# ask of a similar question serves the SAME transient failure forever, even
# long after the underlying provider/network issue has resolved itself.
# Both cache.set() call sites below check this before caching.
_UNCACHEABLE_ANSWERS = {
    ALL_PROVIDERS_DOWN_MESSAGE,
    KB_ONLY_FALLBACK_MESSAGE,
    NO_WEB_RESULTS_MESSAGE,
}


def _is_cacheable(answer: str) -> bool:
    return answer not in _UNCACHEABLE_ANSWERS


# Every node name that can appear in a graph.stream() update, mapped to a
# stable frontend key + human label. Keeping this list explicit (rather than
# titleizing node_name on the fly) means an unrecognized node name from a
# future graph change fails loud in review instead of silently rendering
# a raw snake_case string in the UI.
NODE_STAGE_META = {
    "rewrite_query": {"key": "rewrite", "label": "Rewrite query"},
    "injection_guard": {"key": "injection_guard", "label": "Injection guard"},
    "router": {"key": "router", "label": "Route"},
    "direct_answer": {"key": "direct_answer", "label": "Direct answer"},
    "retrieve": {"key": "retrieve", "label": "Retrieve"},
    "decompose_retrieve": {"key": "decompose_retrieve", "label": "Retrieve (decomposed)"},
    "check_relevance": {"key": "check_relevance", "label": "Check relevance"},
    "grade_documents": {"key": "grade_documents", "label": "Grade documents (CRAG)"},
    "reformulate_query": {"key": "reformulate_query", "label": "Reformulate query"},
    "generate": {"key": "generate", "label": "Generate"},
    "citation": {"key": "citation", "label": "Attach citations"},
    "scope_guard": {"key": "scope_guard", "label": "Scope guard"},
    "out_of_scope": {"key": "out_of_scope", "label": "Web search (Priority 4)"},
    "update_history": {"key": "update_history", "label": "Update history"},
}


def _serialize_node_update(node_name: str, node_output: dict) -> dict:
    """Extracts a small, JSON-safe, frontend-relevant slice of one node's
    state update for the Pipeline Inspector. Deliberately never includes
    full retrieved chunk text or the in-progress answer - only counts,
    source metadata, and routing/guardrail flags."""
    if node_name in ("retrieve", "decompose_retrieve"):
        docs = node_output.get("retrieved_docs", [])
        return {
            "doc_count": len(docs),
            "sources": sorted({d.metadata.get("source", "unknown") for d in docs}),
        }
    if node_name == "out_of_scope":
        # Priority 4: this node now runs a real Tavily web search - surface
        # the same doc_count/sources shape as retrieve, so the Inspector can
        # show "(3 web results)" instead of nothing. An empty docs list here
        # means the fixed-refusal fallback fired (no API key / search failed
        # / zero results), which the frontend can render distinctly.
        docs = node_output.get("retrieved_docs", [])
        return {
            "doc_count": len(docs),
            "sources": [d.metadata.get("source", "unknown") for d in docs],
            "used_web_search": len(docs) > 0,
        }
    if node_name == "check_relevance":
        return {"is_relevant": node_output.get("is_relevant", False)}
    if node_name == "grade_documents":
        return {
            "grading_passed": node_output.get("grading_passed", False),
            "doc_count_after_grading": len(node_output.get("retrieved_docs", [])),
        }
    if node_name == "reformulate_query":
        return {"retry_count": node_output.get("retry_count", 0)}
    if node_name == "router":
        return {
            "route_category": node_output.get("route_category", ""),
            "sub_questions": node_output.get("sub_questions", []),
        }
    if node_name == "injection_guard":
        return {"is_injection": node_output.get("is_injection", False)}
    if node_name == "citation":
        return {"citation_count": len(node_output.get("citations", []))}
    if node_name == "scope_guard":
        return {"scope_flagged": node_output.get("scope_flagged", False)}
    return {}


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

    if _is_cacheable(result["answer"]):
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


def _stream_graph_worker(question: str, thread_id: str, is_new_thread: bool, q: "queue.Queue") -> None:
    """Runs on a background thread. Drives the graph via the SYNCHRONOUS
    graph.stream(stream_mode="updates") - which yields one {node_name:
    state_update} dict per node as it actually completes, real branches and
    reformulate_query loops included, not a fixed 6-step schedule - and
    pushes a small JSON-safe event onto q for each node, then a final event
    with the full answer/citations/pipeline flags, then None as a sentinel.
    Never raises out of this function: any exception is reported as an
    "error" event so the SSE stream can close cleanly instead of hanging."""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = _build_initial_state(question, thread_id, is_new_thread)

        for update in graph.stream(state, config=config, stream_mode="updates"):
            for node_name, node_output in update.items():
                meta = NODE_STAGE_META.get(node_name, {"key": node_name, "label": node_name})
                q.put({
                    "type": "node",
                    "node": meta["key"],
                    "label": meta["label"],
                    "data": _serialize_node_update(node_name, node_output or {}),
                })

        final_state = graph.get_state(config).values
        result = {
            "answer": final_state.get("answer", ""),
            "citations": final_state.get("citations", []),
            "route_category": final_state.get("route_category", ""),
            "is_injection": final_state.get("is_injection", False),
            "scope_flagged": final_state.get("scope_flagged", False),
        }
        if _is_cacheable(result["answer"]):
            cache.set(question, result)
        _fire_feedback_log(thread_id, question, final_state)

        q.put({
            "type": "final",
            "thread_id": thread_id,
            "from_cache": False,
            **result,
        })
    except Exception as exc:
        q.put({"type": "error", "error": str(exc)})
    finally:
        q.put(None)  # sentinel: no more events


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    SSE endpoint powering the frontend's Pipeline Inspector. On a cache hit,
    sends a single honest "cache_hit" event (no real nodes ran, so none are
    faked). On a cache miss, streams one "node" event per actual graph node
    as _stream_graph_worker's background thread produces them, followed by
    one "final" event with the answer/citations. Still not token-by-token
    generation streaming (see module docstring) - this streams pipeline
    STAGE progress in real time, which is what the Inspector needs.
    """
    cached = cache.get(request.question)
    is_new_thread = request.thread_id is None
    thread_id = request.thread_id or str(uuid.uuid4())

    async def event_generator():
        if cached is not None:
            payload = {
                "type": "cache_hit",
                "thread_id": thread_id,
                "from_cache": True,
                "answer": cached["answer"],
                "citations": cached.get("citations", []),
                "route_category": cached.get("route_category", ""),
                "is_injection": cached.get("is_injection", False),
                "scope_flagged": cached.get("scope_flagged", False),
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
            return

        q: "queue.Queue" = queue.Queue()
        threading.Thread(
            target=_stream_graph_worker,
            args=(request.question, thread_id, is_new_thread, q),
            daemon=True,
        ).start()

        loop = asyncio.get_event_loop()
        while True:
            # q.get() blocks - run it on the default executor so it doesn't
            # block this coroutine's event loop between node events.
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _ingest_in_background(thread_id: str, tmp_path: Path, filename: str) -> None:
    """Runs the actual (potentially slow) ingestion pipeline outside any HTTP
    request/response cycle, so there is no timeout ceiling on it anymore.
    Updates _upload_status when done (or on failure) so /upload/status can
    report progress. Always cleans up the temp file, success or failure."""
    try:
        chunk_count = build_session_vectorstore(thread_id, tmp_path, filename)
        with _upload_status_lock:
            _upload_status[thread_id] = {
                "status": "ready",
                "filename": filename,
                "chunks_stored": chunk_count,
                "error": None,
            }
    except Exception as exc:
        logger_msg = str(exc)
        with _upload_status_lock:
            _upload_status[thread_id] = {
                "status": "error",
                "filename": filename,
                "chunks_stored": 0,
                "error": logger_msg,
            }
    finally:
        tmp_path.unlink(missing_ok=True)


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

    FIX: this endpoint now only validates and saves the file, then hands
    off the actual (slow, previously timeout-prone) ingestion work to a
    background thread and returns immediately with status "processing".
    Call GET /upload/status/{thread_id} to poll until ingestion finishes.
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

    with _upload_status_lock:
        _upload_status[thread_id] = {
            "status": "processing",
            "filename": file.filename,
            "chunks_stored": 0,
            "error": None,
        }

    threading.Thread(
        target=_ingest_in_background,
        args=(thread_id, tmp_path, file.filename),
        daemon=True,
    ).start()

    return {
        "thread_id": thread_id,
        "filename": file.filename,
        "status": "processing",
    }


@app.get("/upload/status/{thread_id}")
async def upload_status(thread_id: str):
    """Poll this after /upload to find out when ingestion finishes. Returns
    status: "processing" | "ready" | "error". Frontend should poll every
    2-3s until it sees "ready" or "error", then allow/refuse questions
    against the upload accordingly."""
    with _upload_status_lock:
        status = _upload_status.get(thread_id)
    if status is None:
        raise HTTPException(status_code=404, detail="No upload found for this thread_id")
    return {"thread_id": thread_id, **status}


@app.delete("/upload/{thread_id}")
async def delete_upload(thread_id: str):
    """Removes a thread's uploaded document. Call this when a conversation
    ends, or when the user explicitly wants to remove their uploaded doc
    without ending the whole conversation."""
    await run_in_threadpool(delete_session_vectorstore, thread_id)
    with _upload_status_lock:
        _upload_status.pop(thread_id, None)
    return {"thread_id": thread_id, "status": "deleted"}


@app.get("/health")
async def health():
    return {"status": "ok"}

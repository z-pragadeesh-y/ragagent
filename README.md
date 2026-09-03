# RAGAgent

A Conversational Retrieval-Augmented Generation (RAG) system built with LangGraph, combining hybrid retrieval, agentic query routing, self-correcting retrieval, dynamic document ingestion, source citation, a live web-search fallback, and safety guardrails into a single pipeline — deployed as a real, publicly-reachable web app, not just a local API.

This was built as a learning project to understand how a RAG system evolves from a simple retrieve-and-generate loop into a more structured, agentic pipeline, and then into a deployed product. It answers questions using a fixed local knowledge base covering five domains — AI policy, climate science, economics, public health, and AI research — grounds every answer in retrieved source material rather than the language model's own memorized knowledge, lets users upload their own documents to query alongside (or instead of) that knowledge base, and falls back to a real, clearly-labeled web search rather than a dead-end refusal when a question genuinely falls outside all of the above.

## Live Demo

- **App:** [ragagent-frontend.vercel.app](https://ragagent-frontend.vercel.app/)
- **Backend API:** [ragagent-backend on Google Cloud Run](https://ragagent-backend-643122500993.us-central1.run.app)

The backend runs on Cloud Run's free tier with scale-to-zero, so the very first request after a period of inactivity can take up to a couple of minutes while the container cold-starts (loading the embedding model, reranker, and LLM manager) — subsequent requests are fast. This is an intentional cost trade-off, not a bug.

## 🗺️ Architecture Diagram

An interactive, explorable architecture diagram is available — zoom, pan, trace request routes, and step through guided views of the pipeline (chat flow, hybrid retrieval, LLM failover).

👉 **[View Interactive Architecture Diagram](https://z-pragadeesh-y.github.io/ragagent/architecture.html)**

## Features

- **Hybrid Retrieval** — combines BM25 keyword search and dense vector search using Reciprocal Rank Fusion, then reranks results with a cross-encoder model
- **Agentic Query Routing** — classifies each question before retrieval (direct answer, single-topic retrieval, multi-domain decomposition, or out-of-scope)
- **Corrective Retrieval (CRAG)** — grades retrieved documents for relevance and retries with a reformulated query if the initial retrieval is weak
- **HyDE Query Expansion** — generates a hypothetical answer passage to improve vector search on short or ambiguous questions
- **Structure-Aware Chunking** — splits documents along real section boundaries rather than fixed character counts, and attaches section/domain metadata
- **Dynamic Document Upload** — upload your own PDF/TXT/MD document through the live app and ask questions against it, either alone or combined with the permanent knowledge base in the same conversation; uploaded content is fused into the *same* hybrid retrieval ranking as the permanent corpus, not retrieved separately
- **Web Search Fallback** — questions genuinely outside the knowledge base (confirmed by CRAG, not just an initial routing guess) are answered via a real, live Tavily web search rather than a fixed refusal, always clearly labeled "Based on a web search" and cited the same way as knowledge-base answers
- **Inline Source Citations** — answers include `[Source N]` citations validated against the documents actually retrieved (knowledge base *or* web search), so citations pointing to non-existent sources are detected and removed
- **Basic Guardrails** — a regex + LLM-based check for prompt injection attempts on input, and an LLM-based output check that catches ungrounded opinion/advice drift while still allowing correctly-cited web-sourced answers through
- **Real-Time Pipeline Inspector** — the live frontend streams and displays exactly which pipeline stages ran for each question, in the order they actually ran, via Server-Sent Events — not a fixed animation
- **Feedback Logging** — each query, answer, and guardrail flag is logged to a local SQLite database for later review
- **Multi-Provider LLM Failover** — automatically falls back across Groq, NVIDIA NIM, and an optional local LM Studio model if a provider is unavailable or rate-limited
- **Conversational Memory** — multi-turn conversations are persisted across sessions using LangGraph's SQLite checkpointing
- **Custom Evaluation** — a 30-question golden dataset scored on faithfulness, answer relevancy, context precision, and context recall using an LLM-as-judge approach
- **FastAPI Backend** — a REST API with both a standard and a streaming (SSE) chat endpoint, an async document-upload endpoint with status polling, and semantic response caching (which correctly never caches a transient failure)

## Architecture Overview

```
User Query
    │
    ▼
Query Rewriting (resolves references from prior turns)
    │
    ▼
Injection Guard ──▶ blocked ──▶ fixed refusal
    │
    ▼
Router (direct / retrieve / decompose / out-of-scope — a routing HINT, not a hard gate)
    │
    ▼
Hybrid Retrieval (BM25 + vector search + HyDE) + Reranking
  (fuses the uploaded session document, if any, into the SAME ranking as the permanent KB)
    │
    ▼
Relevance Grading (CRAG) ──▶ if weak ──▶ reformulate query and retry (capped)
    │
    ▼
Answer Generation (with inline citations)
    │
    ├──▶ still no real answer? ──▶ Web Search Fallback (Tavily, clearly labeled)
    │
    ▼
Citation Validation (same validation path for KB or web sources)
    │
    ▼
Scope Guard (checks the answer stays grounded — in its KB or web citations)
    │
    ▼
Response streamed back (live Pipeline Inspector) + feedback logged
```

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM Providers | Groq, NVIDIA NIM, local LM Studio (optional) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector Store | Chroma |
| Keyword Search | BM25 (`rank-bm25`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Web Search | Tavily |
| Backend API | FastAPI |
| Storage | SQLite (conversation state and feedback logs) |
| Frontend | React (Vite), Framer Motion |
| Backend Hosting | Google Cloud Run |
| Frontend Hosting | Vercel |

## Folder Structure

```
graph/                   LangGraph state, nodes, and graph assembly
ingestion/                Document loading, chunking, embedding, hybrid retrieval, HyDE, session (upload) handling
llm/                      Multi-provider LLM manager with failover and task routing
api/                      FastAPI application and semantic caching
eval/                      Golden evaluation set and custom evaluation metrics
feedback/                  Live query/answer logging
data/raw/                  Source knowledge base documents
data/uploaded_markdown/    Persisted, structured markdown for user-uploaded documents
docs/                      Interactive architecture diagram
ragagent-frontend/         React (Vite) frontend — separate GitHub repo, deployed on Vercel
```

## Installation

```bash
git clone https://github.com/z-pragadeesh-y/ragagent.git
cd ragagent

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

The frontend lives in its own repository, [`ragagent-frontend`](https://github.com/z-pragadeesh-y/ragagent-frontend):

```bash
git clone https://github.com/z-pragadeesh-y/ragagent-frontend.git
cd ragagent-frontend
npm install
npm run dev
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your own values:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key for Groq (primary provider for answer generation) |
| `GROQ_MODEL` | Groq model name |
| `NVIDIA_API_KEY` | API key for NVIDIA NIM (used for routing, grading, and query rewriting) |
| `NVIDIA_BASE_URL` | NVIDIA NIM API base URL |
| `NVIDIA_MODEL` | NVIDIA model name |
| `TAVILY_API_KEY` | API key for Tavily (used for the out-of-scope web search fallback) |
| `LM_STUDIO_ENABLED` | `true`/`false` — enables the optional local fallback model |
| `LM_STUDIO_BASE_URL` | Local LM Studio server URL (only used if enabled) |
| `LM_STUDIO_MODEL` | Local model name in LM Studio |
| `LLM_MAX_RETRIES_PER_PROVIDER` | Retry attempts before failing over to the next provider |
| `LLM_RETRY_BACKOFF_SECONDS` | Delay between retries |
| `LLM_LOG_TO_FILE` | `true`/`false` — enables logging to a file |
| `LLM_LOG_FILE_PATH` | Log file path |

## Running the Project

Run the regression test suite:

```bash
python -m graph.test_cases
```

Start the API server:

```bash
uvicorn api.main:app --reload --reload-dir api --reload-dir graph --reload-dir llm --reload-dir ingestion --reload-dir feedback
```

The `--reload-dir` flags scope the auto-reloader to source code only — without them, `uvicorn --reload`'s default watcher also watches `checkpoints.sqlite` and `.chroma/`, both written to on every request, causing spurious full-server restarts mid-session.

The server runs at `http://localhost:8000` by default.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Submit a question, receive the full answer and citations at once |
| `POST` | `/chat/stream` | Same as above, delivered as a Server-Sent Events stream — including live, per-node pipeline progress events, not just the final answer |
| `POST` | `/upload` | Upload a document (PDF/TXT/MD) for this conversation thread; returns immediately with `{"status": "processing"}` |
| `GET` | `/upload/status/{thread_id}` | Poll ingestion status for an uploaded document (`processing` / `ready` / `error`) |
| `DELETE` | `/upload` | Clear the uploaded session document for this conversation thread |
| `GET` | `/health` | Basic health check |

Example request:

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"question\": \"What does the IMF forecast for global economic growth?\"}"
```

## Example Queries

- "What is the AI Risk Management Framework meant to help organizations do?"
- "What are the key findings of the IPCC AR6 report on climate change?"
- "How does climate change affect public health outcomes?"
- "What is Retrieval-Augmented Generation and why is it used with LLMs?"

Questions outside the five covered domains (AI policy, climate, economics, health, AI research) — e.g. "What's the weather right now?" or "What is Breaking Bad?" — are correctly identified as out-of-scope and answered via a real, clearly-labeled web search instead of a hard refusal or a hallucinated guess.

## Known Limitations

- Uploading a large document (roughly 900KB+) currently times out during ingestion. Smaller documents (tested up to several hundred KB) work correctly. This is a known, deferred gap requiring a real background-job/queue architecture rather than a prompt-level fix.
- The `/chat/stream` endpoint streams real, live pipeline *stage* progress (which node ran, with what result) but not token-by-token answer generation — the final answer text itself still arrives as one event once generation completes.
- Cloud Run's `--max-instances=1` + scale-to-zero configuration means the first request after a period of inactivity is slow (cold start); this is an accepted cost trade-off, not a bug.
- The backend currently has no rate limiting on its public endpoints.

## Future Improvements

- True token-by-token answer streaming
- A background job queue for document ingestion, to properly support large uploads
- A user-facing thumbs-up/down feedback mechanism, building on the existing logging infrastructure
- Expanding the permanent knowledge base beyond the current five source documents
- Rate limiting / basic abuse protection on the public backend

## License

MIT — see [LICENSE](./LICENSE).

## Project Background

This project was built incrementally and documented in detail as it progressed. See [`DEVLOG.md`](./DEVLOG.md) for the full build history and every bug encountered along the way, and [`LEARNING_NOTES.md`](./LEARNING_NOTES.md) for the underlying concepts behind each component.

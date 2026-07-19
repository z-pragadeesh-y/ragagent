# RAGAgent

A Conversational Retrieval-Augmented Generation (RAG) system built with LangGraph, combining hybrid retrieval, agentic query routing, self-correcting retrieval, source citation, and basic safety guardrails into a single pipeline.

This was built as a learning project to understand how a RAG system evolves from a simple retrieve-and-generate loop into a more structured, agentic pipeline. It answers questions using a fixed local knowledge base covering five domains — AI policy, climate science, economics, public health, and AI research — and grounds every answer in retrieved source material rather than relying on the language model's own memorized knowledge.

## Features

- **Hybrid Retrieval** — combines BM25 keyword search and dense vector search using Reciprocal Rank Fusion, then reranks results with a cross-encoder model
- **Agentic Query Routing** — classifies each question before retrieval (direct answer, single-topic retrieval, multi-domain decomposition, or out-of-scope)
- **Corrective Retrieval (CRAG)** — grades retrieved documents for relevance and retries with a reformulated query if the initial retrieval is weak
- **HyDE Query Expansion** — generates a hypothetical answer passage to improve vector search on short or ambiguous questions
- **Structure-Aware Chunking** — splits documents along real section boundaries rather than fixed character counts, and attaches section/domain metadata
- **Inline Source Citations** — answers include `[Source N]` citations validated against the documents actually retrieved, so citations pointing to non-existent sources are detected and removed
- **Basic Guardrails** — a regex + LLM-based check for prompt injection attempts on input, and an LLM-based scope check on output to catch answers that drift into unsupported opinion
- **Feedback Logging** — each query, answer, and guardrail flag is logged to a local SQLite database for later review
- **Multi-Provider LLM Failover** — automatically falls back across Groq, NVIDIA NIM, and an optional local LM Studio model if a provider is unavailable or rate-limited
- **Conversational Memory** — multi-turn conversations are persisted across sessions using LangGraph's SQLite checkpointing
- **Custom Evaluation** — a 30-question golden dataset scored on faithfulness, answer relevancy, context precision, and context recall using an LLM-as-judge approach
- **FastAPI Backend** — a REST API with both a standard and a streaming (SSE) chat endpoint, plus basic semantic response caching

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
Router (direct / retrieve / decompose / out-of-scope)
    │
    ▼
Hybrid Retrieval (BM25 + vector search + HyDE) + Reranking
    │
    ▼
Relevance Grading ──▶ if weak ──▶ reformulate query and retry
    │
    ▼
Answer Generation (with inline citations)
    │
    ▼
Citation Validation
    │
    ▼
Scope Guard (checks the answer stays grounded)
    │
    ▼
Response returned + feedback logged
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
| Backend API | FastAPI |
| Storage | SQLite (conversation state and feedback logs) |

## Folder Structure

```
graph/       LangGraph state, nodes, and graph assembly
ingestion/   Document loading, chunking, embedding, hybrid retrieval, HyDE
llm/         Multi-provider LLM manager with failover and task routing
api/         FastAPI application and semantic caching
eval/        Golden evaluation set and custom evaluation metrics
feedback/    Live query/answer logging
data/raw/    Source knowledge base documents
```

## Installation

```bash
git clone https://github.com/z-pragadeesh-y/CONVERSATIONAL--RAGAGENT.git
cd CONVERSATIONAL--RAGAGENT

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
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
uvicorn api.main:app --reload
```

The server runs at `http://localhost:8000` by default.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Submit a question, receive the full answer and citations at once |
| `POST` | `/chat/stream` | Same as above, delivered as a Server-Sent Events stream |
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

Questions outside the five covered domains (AI policy, climate, economics, health, AI research) are correctly identified as out-of-scope and answered with a refusal rather than a guess.

## Future Improvements

- True token-by-token streaming (current streaming endpoint returns the full answer as a single event, not incremental tokens)
- A hosted, deployed version with a simple web frontend
- Expanding the knowledge base beyond the current five source documents
- A user-facing thumbs-up/down feedback mechanism, building on the existing logging infrastructure

## License

MIT — see [LICENSE](./LICENSE).

## Project Background

This project was built incrementally and documented in detail as it progressed. See [`DEVLOG.md`](./DEVLOG.md) for the full build history and every bug encountered along the way, and [`LEARNING_NOTES.md`](./LEARNING_NOTES.md) for the underlying concepts behind each component.

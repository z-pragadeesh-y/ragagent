# RAGAgent

An agentic, self-correcting Retrieval-Augmented Generation (RAG) system built with LangGraph — combining hybrid retrieval, query routing, corrective retrieval, safety guardrails, and verifiable source citations into a single production-style pipeline.

Answers are grounded strictly in a local knowledge base spanning five domains (AI policy, climate science, economics, public health, and AI research) — the system never relies on an LLM's own memorized knowledge, and every claim it makes is traceable back to a real retrieved source.

## Features

- **Hybrid Retrieval** — combines BM25 (keyword) and dense vector search via Reciprocal Rank Fusion, then reranks candidates with a cross-encoder for high-precision results
- **Agentic Query Routing** — classifies each question (direct answer / single-topic retrieval / multi-domain decomposition / out-of-scope) before deciding how to handle it
- **Corrective RAG (CRAG)** — grades retrieved documents for relevance and self-corrects by reformulating the query and retrying, rather than answering from poor context
- **HyDE Query Expansion** — generates a hypothetical answer passage to improve vector search recall on short or awkwardly-phrased questions
- **Structure-Aware Chunking** — splits source documents on real section boundaries with rich metadata, instead of blind fixed-size chunking
- **Verifiable Citations** — every answer cites its sources inline (`[Source N]`), validated against real retrieved chunks to prevent hallucinated citations
- **Guardrails** — input-side prompt-injection detection (regex + LLM fallback) and output-side scope enforcement, so answers stay grounded and on-topic
- **Feedback Logging** — every query, answer, citation set, and guardrail flag is logged to SQLite for observability
- **Multi-Provider LLM Failover** — automatic failover across Groq, NVIDIA NIM, and a local LM Studio model, with task-based routing between a fast/cheap lane and a high-quality lane
- **Conversational Memory** — multi-turn conversations with persistent, checkpointed state (SQLite-backed)
- **Evaluation Suite** — custom RAGAS-style metrics (faithfulness, answer relevancy, context precision, context recall) run against a 30-question golden dataset
- **API Layer** — FastAPI backend with standard and streaming (SSE) chat endpoints, plus semantic response caching

## Architecture

```
User Query
    │
    ▼
Query Rewriting (resolves conversational references)
    │
    ▼
Injection Guard (regex + LLM fallback) ──▶ blocked ──▶ refusal
    │
    ▼
Agentic Router
    │
    ├── direct answer (greetings/meta)
    ├── single-topic retrieval
    ├── multi-domain decomposition
    └── out-of-scope (refusal)
    │
    ▼
Hybrid Retrieval (BM25 + vector + HyDE) + Reranking
    │
    ▼
Corrective Grading ──▶ (fail) ──▶ reformulate & retry
    │
    ▼
Answer Generation (with inline citations)
    │
    ▼
Citation Validation
    │
    ▼
Scope Guard (LLM-based output check)
    │
    ▼
Response + Feedback Logging
```

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (stateful graph execution) |
| LLM Providers | Groq, NVIDIA NIM, local LM Studio (via LangChain) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector Store | Chroma |
| Keyword Search | BM25 (`rank-bm25`) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| API | FastAPI |
| Persistence | SQLite (conversation checkpoints + feedback logs) |

## Getting Started

### Prerequisites
- Python 3.10+
- API keys for [Groq](https://console.groq.com) and [NVIDIA NIM](https://build.nvidia.com) (both have free tiers)
- (Optional) [LM Studio](https://lmstudio.ai) running locally, for an offline fallback provider

### Installation

```bash
git clone https://github.com/z-pragadeesh-y/CONVERSATIONAL--RAGAGENT.git
cd CONVERSATIONAL--RAGAGENT
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and fill in your own API keys:

```bash
cp .env.example .env
```

### Running

Run the test suite to verify everything is working:

```bash
python -m graph.test_cases
```

Start the API server:

```bash
uvicorn api.main:app --reload
```

Then send a request:

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"question\": \"What does the IMF forecast for global economic growth?\"}"
```

## Project Structure

```
graph/       — LangGraph nodes, state, and graph assembly
ingestion/   — document loading, chunking, embedding, hybrid retrieval, HyDE
llm/         — multi-provider LLM manager with failover and task-based routing
api/         — FastAPI app, semantic caching
eval/        — golden evaluation set + custom RAGAS-style metrics
feedback/    — live usage logging
data/raw/    — source knowledge base documents
```

## Evaluation

The system is evaluated against a 30-question golden dataset spanning all five knowledge domains, scored on faithfulness, answer relevancy, context precision, and context recall using LLM-as-judge methodology. See `eval/golden_set.py` and `eval/custom_ragas.py`.

## Project Journal

This project was built incrementally, in verified stages, with every design decision, bug, and fix documented in detail — see [`DEVLOG.md`](./DEVLOG.md) for the full build history, and [`LEARNING_NOTES.md`](./LEARNING_NOTES.md) for the underlying concepts and theory behind each component.

## License

MIT

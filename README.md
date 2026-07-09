# RAGAgent

## Project Title
**RAGAgent** — A Progressive Conversational Retrieval-Augmented Generation (RAG) Agent

## What This Project Does
RAGAgent is a hands-on learning project that builds a conversational RAG system **from scratch, in stages of increasing sophistication** — starting from a bare-bones "Vanilla RAG" pipeline and progressively evolving into a full **Agentic RAG** system with hybrid retrieval, query routing, self-correction, evaluation, and trust/safety features.

The system answers natural-language questions by retrieving relevant passages from a fixed local knowledge base (5 domain documents spanning AI policy, climate science, economics, public health, and AI research) and generating grounded answers using an LLM — never relying on the LLM's own memorized knowledge alone.

**Primary goal of this project is learning**, not just shipping a working app. Each phase is built, verified, and understood conceptually (theory + code) before moving to the next. Known limitations at each stage are deliberately left in place if a later phase is designed to fix them, so real "before vs after" comparisons can be made.

## LLM Provider
**Groq API** (via `langchain-groq`) is used for **all** LLM generation calls throughout the entire project. Currently using model `llama-3.3-70b-versatile`.

## Embeddings Policy
All embedding models are **free and local** (no API cost) — currently `sentence-transformers/all-MiniLM-L6-v2` (384-dim, runs on CPU via HuggingFace). This policy holds for the whole project; only LLM generation goes through Groq's API.

## Project Structure (phases / layers)
The project is built in two broad plans, sitting on top of an initial environment setup phase:

- **Phase 0 — Environment Setup**: folders, venv, git, dependencies, API key config. No RAG logic.
- **Phase 1 — Vanilla RAG**: the simplest possible working RAG pipeline (load → chunk → embed → store → retrieve → generate). No agentic behavior, no query routing, fixed-size "dumb" chunking.
- **Phase 2 — LangGraph State Machine** *(planned)*: convert the linear retrieve→generate chain into a graph-based, stateful flow using LangGraph — the foundation for later agentic behavior.
- **Phase 3 — Query Rewriting** *(planned)*: improve raw user queries before retrieval.
- **Phase 4 — Persistence (Redis/PostgreSQL)** *(planned)*: add conversational memory / session persistence.

On top of Phases 0–4, two enhancement plans will be layered in, per `RAG_Project_Roadmap_v2.docx`:

- **Plan 1 (pipeline & reasoning quality)**:
  - Step 1: Hybrid Retrieval + Reranking (BM25 + vector search + Reciprocal Rank Fusion + cross-encoder reranker)
  - Step 2: Agentic Router (4-way classification: direct / simple retrieval / decompose / out-of-scope)
  - Step 3: Corrective Retrieval Loop (CRAG-style self-correction when retrieval quality is poor)
  - Step 4: RAGAS Evaluation (automated retrieval + generation quality metrics, non-proportional golden set across the 5 domains)
  - Step 5: Streaming responses + semantic caching

- **Plan 2 (data quality & trust)**:
  - Step 6: Ingestion Overhaul — semantic/structure-aware chunking, contextual retrieval, activating the `domain_tag`/`section_title` metadata already present in source docs (currently unused by Phase 1's dumb chunking), regex cleanup of PDF-extraction artifacts
  - Step 7: HyDE (Hypothetical Document Embeddings) query expansion
  - Step 8: Citation & source attribution in generated answers
  - Step 9: Guardrails (input/output safety, scope enforcement)
  - Step 10: Feedback logging for continuous improvement

## Data Description
The knowledge base consists of **5 markdown documents** in `data/raw/`, each derived from a real public-domain source PDF (originally merged into one PDF, then split and cleaned), and each carrying YAML frontmatter metadata (`domain_tag`, `section_title`, `section_number`, `parent_section`, etc.) that is **not yet used** by Phase 1 but is reserved for Plan 2's structure-aware chunking:

| File | Domain (`domain_tag`) | Source | Approx. size |
|---|---|---|---|
| `document_1_AI_optimized.md` | `ai_policy` | NIST AI Risk Management Framework (AI RMF 1.0), 2023 | ~116k chars / ~16.9k words |
| `document_2_Climate_optimized.md` | `climate` | IPCC AR6 Synthesis Report — Summary for Policymakers | ~148k chars / ~20.7k words |
| `document_3_Economics_optimized.md` | `economics` | IMF World Economic Outlook, April 2026, Ch. 1 | ~186k chars / ~28.8k words |
| `document_4_Health_optimized.md` | `health` | WHO World Health Statistics 2025 | ~226k chars / ~33k words |
| `document_5_Natural_optimized.md` | `ai_research` | Retrieval-Augmented Generation for Large Language Models — Survey (arXiv) | ~114k chars / ~15.9k words |

Data was verified for fidelity against the original source PDF (spot-checked facts, confirmed correct document boundaries, confirmed table-of-contents blocks were properly stripped). One known, deliberately-untouched artifact: a few mid-word hyphenation breaks in the Health doc from PDF extraction — left as-is intentionally, to be cleaned in Plan 2 Step 6 for a genuine before/after quality comparison.

## How This Is Being Built
Each phase follows the same loop:
1. Concept is explained (theory) before any code is written
2. Code is written incrementally, one script/module at a time
3. Each script is run and its output is manually verified against an expected-output description (not just "does it run" but "is the result actually correct")
4. Known limitations surfaced during testing are documented, not silently patched, if a later phase is designed to address them
5. Progress is logged in this file after each phase completes

---

## Progress Log

### Phase 0 — Environment Setup ✅ Complete
**What was done:**
- Created project root at `E:\project1\ragagent`
- Created folder structure: `api/`, `data/raw/`, `eval/`, `graph/`, `ingestion/`, `tests/`
- Created a Python virtual environment (`venv`) and activated it
- Initialized a git repository, created `.gitignore`, made first commit
- Placed the 5 domain markdown documents into `data/raw/`
- Decided on Groq as the sole LLM provider for the project
- Created and populated `.env` with `GROQ_API_KEY`

**New files created (Phase 0):**
- `.env` — holds `GROQ_API_KEY` (excluded from git)
- `.gitignore` — excludes `venv/`, `.env`, `__pycache__/`, `.chroma/`
- `requirements.txt` — Python dependency list
- `README.md` — this file
- `main.py` — placeholder at this stage, later became the Phase 1 entry point

**Why:** Establishes a clean, reproducible, version-controlled base before any RAG logic is written, so later phases can be built without revisiting setup concerns.

---

### Phase 1 — Vanilla RAG ✅ Complete
**What was done:**
Built the simplest possible end-to-end RAG pipeline: load documents → split into fixed-size chunks → embed chunks locally → store in a vector database → retrieve relevant chunks for a query → generate a grounded answer with Groq. Verified each step's output manually before moving to the next (5 docs loaded correctly, ~1193 chunks created proportional to doc sizes, 384-dim embeddings confirmed local/free, semantic search returned domain-correct results, and the full chain produced a grounded answer).

Also surfaced and documented a real limitation: fixed-size chunking can separate content from its document-level framing (a NIST-specific question initially failed because the retrieved chunk didn't repeat the word "NIST", even though it was clearly NIST content) — the LLM correctly declined to guess rather than hallucinate. This is a known Vanilla RAG weakness, intentionally left unfixed here since Plan 2 Step 6 (structure-aware chunking) is designed to address it later.

**New files created (Phase 1), and why:**
- `ingestion/loader.py` — reads all `.md` files from `data/raw/` into LangChain `Document` objects; the entry point of the ingestion pipeline
- `ingestion/chunker.py` — splits loaded documents into fixed-size overlapping chunks (baseline "dumb" chunking strategy, deliberately simple to serve as a comparison baseline for later smarter chunking)
- `ingestion/embedder.py` — loads a local, free HuggingFace embedding model and generates vector embeddings for chunks, with no API cost
- `ingestion/vectorstore.py` — builds and persists a local Chroma vector database from the embedded chunks, and provides semantic similarity search
- `ingestion/__init__.py`, `graph/__init__.py`, `api/__init__.py`, `eval/__init__.py` — empty package marker files, added so each project folder can be correctly imported as a Python module (needed once `ingestion.loader` etc. started being imported across files)
- `main.py` (rewritten) — the Phase 1 entry point: ties retrieval (Chroma) and generation (Groq LLM) together into the first working retrieve-then-generate RAG chain

**Result:** A working Vanilla RAG system, fully local except for the final Groq generation call, forming the baseline that every later phase and plan step will be measured against.

---

### Phase 2 — LangGraph State Machine ✅ Complete (verified after fixes)
**What was done:**
Refactored Phase 1's linear retrieve→generate chain into a **LangGraph state machine** — a shared state object (`RAGState`) flowing through discrete nodes connected by edges. First confirmed the refactor preserved Phase 1's exact behavior (identical output for the same test question), proving the restructuring introduced no regressions. Then added a **relevance-check node with a conditional edge**: after retrieval, the graph now checks whether retrieved chunks are actually close enough to the query to bother generating an answer, and routes to either `generate` or a fixed `out_of_scope` response — skipping the LLM call entirely for clearly out-of-scope questions.

While building the relevance check, caught and fixed a real bug: `similarity_search_with_relevance_scores` was not returning true 0–1 normalized scores as its name implied (a `UserWarning` exposed this) — switched to `similarity_search_with_score` (raw L2 distance, where lower = more similar) with a distance-based threshold instead.

**Verification round (10-question test suite across all 5 domains + edge cases):**
Ran a batch of 10 test questions — one per domain, several clearly out-of-scope questions, and two deliberately broad/cross-domain questions — through the full graph. Initial run: 10/10 correct on relevance *classification*, but 2 answers contradicted their own relevance flag (marked relevant, correct source retrieved, yet the LLM still replied "I don't have enough information"). Debugged directly by inspecting raw retrieved chunk content: for broad questions like "what are the key findings," the top-scoring chunks were meta/structural text (citation blocks, section titles, author lists) rather than substantive content, because framing words ("findings," "risks") matched those chunks too, and `k=4` wasn't deep enough to also surface the real content chunks.

**Fix applied:** increased retrieval depth from `k=4` to `k=8`, and loosened the generation prompt to explicitly instruct synthesis across partial/spread-out context, rather than requiring an exact literal match before answering. Re-ran the full 10-question suite: 10/10 passed with genuinely correct, non-contradictory answers.

**Important honesty note (not glossed over):** `k=8` + prompt loosening is a real, working fix — but it is a **band-aid**, not a structural solution. It works by brute-force retrieving more chunks so a relevant one is statistically more likely to be included, and by asking the LLM to try harder to synthesize from noisy context. The actual robust fix — ensuring the *right* chunks are ranked highest in the first place — is Plan 1 Step 1 (hybrid retrieval: BM25 + vector search + reranking). This will be directly tested against the current baseline later using the same 10-question suite.

**New files created (Phase 2), and why:**
- `graph/state.py` — defines `RAGState`, the shared typed state object (question, retrieved_docs, answer, is_relevant) that flows through every node in the graph
- `graph/nodes.py` — the graph's node functions: `retrieve_node`, `check_relevance_node`, `generate_node`, `out_of_scope_node`, and the conditional routing function `route_after_relevance_check`
- `graph/build_graph.py` — assembles all nodes and edges (including the conditional edge) into a compiled, runnable LangGraph state machine; also the manual entry point for quick single-question testing
- `graph/debug_retrieval.py` — diagnostic script to inspect raw retrieved chunk content and distance scores for a given question, used to root-cause the contradictory-answer bug
- `graph/test_cases.py` — a repeatable 10-question verification suite spanning all 5 domains plus edge cases, used to catch and confirm the fix for the retrieval-depth bug; kept in the project as a regression check for future phases

**Result:** A graph-structured RAG pipeline with basic conditional routing (relevant vs. out-of-scope), verified correct across all 5 domains and multiple edge cases, with one honestly-documented shallow-retrieval limitation still pending a proper fix in Plan 1.

---

*(This section will be extended after each subsequent phase completes.)*

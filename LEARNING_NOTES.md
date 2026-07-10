# RAGAgent — Learning Notes

Personal study companion for the RAGAgent project. Each phase gets a section covering: what each file does, why it exists, the theory behind it, and related concepts worth understanding deeply. Code is referenced by file/function name only — the actual code lives in the project files.

---

## Phase 0 — Environment Setup

### Concepts

**Virtual environments (`venv`)**
A `venv` is an isolated Python installation just for this project. Without it, every package you `pip install` goes into your global Python, and different projects can silently conflict (Project A needs `langchain==0.1`, Project B needs `langchain==0.3` — global installs can't satisfy both). Activating a venv (`venv\Scripts\activate`) makes `python` and `pip` point at this isolated copy until you close the terminal or deactivate.

**`.gitignore`**
Tells git which files/folders to never track. We exclude:
- `venv/` — huge, machine-specific, regeneratable from `requirements.txt`
- `.env` — contains secrets (API keys); committing this would leak your Groq key if the repo is ever public
- `__pycache__/` — Python's compiled bytecode cache, regenerated automatically
- `.chroma/` — the vector database, large and fully regeneratable from source docs

**Environment variables & `.env` files**
Hardcoding an API key directly in code is a security risk (easy to accidentally commit/leak). Instead, secrets live in a `.env` file (git-ignored), and `python-dotenv`'s `load_dotenv()` reads that file into `os.environ` at runtime, so code accesses it via `os.getenv("GROQ_API_KEY")` without the key ever appearing in source code.

**Why Python packages need `__init__.py`**
Historically, an empty `__init__.py` file is what tells Python "this folder is a package, not just a directory" — enabling imports like `from ingestion.loader import load_documents` across files. (Modern Python can sometimes infer this via "namespace packages," but explicit `__init__.py` avoids import errors and is the more portable/predictable choice, which is why we hit a `ModuleNotFoundError` until adding them.)

---

## Phase 1 — Vanilla RAG

### The Big Picture: What is RAG?

An LLM alone only "knows" what was in its training data — frozen at some cutoff, and it can't cite specific private/custom documents. **Retrieval-Augmented Generation (RAG)** fixes this by, at query time: (1) searching your own document collection for relevant passages, (2) inserting those passages into the LLM's prompt as "context," and (3) instructing the LLM to answer using that context rather than (only) its internal knowledge. This grounds answers in real, verifiable source material and lets you update the knowledge base without retraining anything.

"Vanilla" RAG = the simplest possible version of this loop, with no smart routing, no query rewriting, no reranking, no self-correction — just retrieve-then-generate. It's the baseline every more advanced RAG technique improves upon.

---

### `ingestion/loader.py` — Document Loading

**What it does:** Reads every `.md` file from `data/raw/` and wraps each one in a LangChain `Document` object — a simple container with `page_content` (the raw text) and `metadata` (a dict, here just `{"source": filename}`).

**Why a wrapper object instead of raw strings?** Every downstream LangChain component (splitters, vector stores, retrievers) expects this standard `Document` shape. Keeping metadata attached to content means that even after splitting a document into 300 tiny chunks, each chunk still "remembers" which file it came from — critical for later steps like source citation.

**Concept: separation of concerns.** Loading is kept as its own step/file rather than jammed into the vector store code. Each ingestion stage (load → chunk → embed → store) is a distinct function you can test, swap out, or debug independently — a common software engineering pattern that matters a lot in ML pipelines where any one stage can silently produce bad data.

---

### `ingestion/chunker.py` — Chunking

**What it does:** Splits each loaded `Document` into smaller `Document` chunks of ~1000 characters, using `RecursiveCharacterTextSplitter` with a 200-character overlap between consecutive chunks.

**Why chunk at all?**
1. Embedding models have a maximum input length — you can't embed a 200,000-character document as a single vector meaningfully; the vector would be a blurry average of everything, useless for precise retrieval.
2. LLMs also have context window limits — you can't stuff whole documents into every prompt.
3. Smaller, focused chunks retrieve more precisely: a query about "premature mortality" should match a chunk specifically about that, not get diluted by an entire 33,000-word health report.

**Why `RecursiveCharacterTextSplitter` specifically?** It tries to split on natural boundaries first (paragraph breaks, then sentences, then words) before falling back to a hard character cut — "recursive" in the sense that it recursively tries larger separators first, then smaller ones, to keep chunks as semantically coherent as possible while still respecting the size limit. Even so, in Phase 1 this is still considered a "dumb"/naive strategy compared to true *semantic chunking* (splitting based on meaning shifts, not just character count) — that's a Phase-2-plan-later upgrade.

**Why overlap (200 chars)?** If an important sentence happens to fall right at a chunk boundary, without overlap it could get cut in half and lose meaning in both resulting chunks. Overlap ensures boundary content appears intact in at least one chunk.

**The trade-off you should understand:** Smaller chunks → more precise retrieval but less surrounding context per chunk (this is literally what caused the NIST-attribution failure you saw — the chunk had the right content but not the right "framing" word). Larger chunks → more context but noisier, less precise retrieval. Chunk size is a real tuning parameter, not an arbitrary constant.

---

### `ingestion/embedder.py` — Embeddings

**What it does:** Loads a local, free sentence-embedding model (`sentence-transformers/all-MiniLM-L6-v2`) that converts any text string into a 384-number vector.

**Core concept: embeddings represent meaning as geometry.** Two pieces of text with similar meaning get mapped to vectors that are close together in this 384-dimensional space (measured by cosine similarity or dot product); unrelated text ends up far apart. This is fundamentally different from keyword search — "car" and "automobile" could be close in embedding space despite sharing zero letters.

**Why this specific model?** `all-MiniLM-L6-v2` is a widely-used, small (~80MB), fast, CPU-friendly sentence-transformer — a good default baseline for learning. It's not the most powerful embedding model available, but it's free, local, and "good enough" to prove the pipeline works. This is itself a real engineering decision: bigger/better embedding models exist (some via paid APIs), and swapping this out later is a legitimate optimization path once you understand the baseline.

**`normalize_embeddings=True`** — scales every vector to unit length (length 1). This makes cosine similarity and dot-product similarity mathematically equivalent, simplifying and slightly speeding up the similarity math the vector store does later.

**Why local instead of an API-based embedding model (e.g., OpenAI's)?** No cost, no rate limits, no network dependency, and — deliberately, per your own constraint — this keeps the "free tier" of the project as large as possible, reserving the Groq API budget purely for the LLM generation step where it's actually needed.

---

### `ingestion/vectorstore.py` — Vector Database (Chroma)

**What it does:** Takes all ~1193 chunk `Document`s, embeds each one using the embedder, and stores the resulting vectors (plus original text and metadata) in **Chroma**, a local, file-based vector database, persisted to a `.chroma/` folder on disk.

**Why do we need a database instead of just a list of vectors in memory?**
1. Persistence — you don't want to re-embed 1193 chunks every time you run the app; Chroma saves them to disk once and loads them back instantly afterward.
2. Efficient similarity search — Chroma implements optimized nearest-neighbor search algorithms so that finding "the 3 closest vectors to my query vector" is fast even as chunk count grows, rather than a slow brute-force Python loop.

**Core concept: similarity search / nearest-neighbor search.** When you call `similarity_search(query, k=3)`, Chroma: (1) embeds your query text using the same embedding model, (2) computes how close that query vector is to every stored chunk vector, (3) returns the `k` closest chunks. This is the literal mechanism behind "semantic search" — no keywords are matched, only vector proximity.

**Why does this matter for verification?** When you tested the query "What is climate change causing globally?" and got back 3 chunks all correctly from the Climate document — that's direct proof the embedding model is genuinely capturing topic/meaning, not just returning arbitrary chunks. This is the crucial sanity check any RAG system needs before trusting it further.

---

### `main.py` — The Retrieve-and-Generate Chain

**What it does:** Ties everything together. For a given question: (1) retrieves the top-`k` (here, 4) most relevant chunks from Chroma, (2) formats them into a single "context" string tagged with their source filenames, (3) inserts that context plus the question into a prompt template, (4) sends the filled prompt to Groq's LLM (`llama-3.3-70b-versatile`), (5) returns the generated answer plus which source chunks were used.

**Concept: prompt engineering for grounding.** The prompt explicitly instructs: *"answer using ONLY the provided context... if the answer is not in the context, say you don't have enough information."* This single instruction is what separates RAG from just asking an LLM a question directly — it forces the model to lean on retrieved evidence and refuse to fabricate an answer when the evidence doesn't support one. You directly observed this behavior: when the retrieved chunks didn't literally contain "NIST," the model correctly refused rather than guessing — this is the grounding instruction working as designed, not a malfunction.

**Concept: `k` (top-k retrieval).** `k=4` means "retrieve the 4 most relevant chunks." Too small a `k` risks missing needed information; too large a `k` risks diluting the prompt with irrelevant/noisy chunks (and costs more tokens). This is another real tuning parameter — Plan 1's Step 1 (hybrid retrieval + reranking) exists specifically to make retrieval smarter than a flat top-k similarity search.

**Concept: LangChain's chain composition (`prompt | llm`).** The `|` operator here is LangChain's "Runnable" pipe syntax — it means "take the prompt template's output and feed it directly into the LLM as input," forming a small pipeline. This composability becomes essential later in Phase 2, when these small chains get embedded as **nodes** inside a larger LangGraph state machine.

**Why Groq specifically?** Groq provides very fast LLM inference (their custom hardware, "LPUs," is optimized for low-latency generation) with an API compatible with LangChain's standard `ChatModel` interface — good for fast iteration while learning, and it's the provider chosen for the whole project per your requirement.

---

### Key Takeaway From Phase 1 (worth remembering)

Vanilla RAG's biggest structural weakness: **retrieval quality is entirely dependent on chunk-level semantic similarity, with zero awareness of document structure, named entities not repeated in-chunk, or query intent.** The NIST example is a perfect textbook case: the *content* needed was retrieved correctly (right document, right topic), but the *specific term* the question used ("NIST") wasn't present in the top chunks verbatim, and there was no mechanism to reformulate the query or expand context. Every subsequent phase/step in this project (hybrid retrieval, reranking, HyDE query expansion, agentic routing, structure-aware chunking) is, in one way or another, a fix for this exact class of problem. Keep this example in mind — it's the cleanest possible illustration of *why* more advanced RAG techniques exist.

---

## Phase 2 — LangGraph State Machine

### The Big Picture: Why Graphs Instead of Straight-Line Code?

Phase 1's `main.py` was a function calling a function: `retrieve() → generate()`. That works when there's exactly one path through the logic. But real RAG systems need to **branch** ("is this even answerable from my data?"), **loop** ("retrieval was bad, try again with a reformulated query"), and **hold evolving state** across many steps (routing decisions, retry counts, conversation history). You can't cleanly express "if X then go here, else go there, and loop back sometimes" with plain function calls without it turning into unreadable nested if/else spaghetti fast.

**LangGraph** solves this by modeling your pipeline as an explicit **graph**: a set of **nodes** (units of work) connected by **edges** (control flow, which can be conditional). This isn't just a stylistic choice — it's a structural prerequisite for everything coming later (Plan 1's agentic router, correction loop) to be added *without rewriting the whole pipeline*.

---

### `graph/state.py` — The State Object

**What it does:** Defines `RAGState`, a `TypedDict` with fields: `question`, `retrieved_docs`, `answer`, `is_relevant`.

**Core concept: shared mutable state.** In LangGraph, every node receives the *entire current state* and returns a partial update (a dict with just the keys it changed). LangGraph merges that update into the state automatically before passing it to the next node. This means nodes don't need to know about each other directly — they only need to agree on the shape of the shared state. This is what makes nodes swappable/composable: you could replace `retrieve_node` entirely with a hybrid-retrieval version later (Plan 1 Step 1) without touching `generate_node` at all, as long as it still fills in `retrieved_docs` correctly.

**Why `TypedDict` specifically?** It gives you a documented, type-checked schema for the state (your editor/type-checker can catch typos like `state["qeustion"]`), while still being a plain dict at runtime — which is what LangGraph expects.

---

### `graph/nodes.py` — Nodes as Pure(ish) Functions

**What it does:** Contains four node functions plus one routing function.

**`retrieve_node`** — same logic as Phase 1's retrieval step, just now shaped as `(state) -> partial_state_update` instead of returning a value directly. Note it now retrieves `k=8` chunks (increased from the original `k=4` — see the "verification and fix" section below for why).

**`check_relevance_node`** — new in Phase 2. Calls Chroma's `similarity_search_with_score`, which returns `(document, distance)` pairs. **Important nuance learned here:** Chroma's *default* distance metric is L2 (Euclidean) distance, where **lower = more similar** — the opposite intuition from a 0–1 "similarity score" where higher = better. We initially used `similarity_search_with_relevance_scores` assuming it would give a clean 0–1 similarity score, but it threw a `UserWarning` because the actual returned numbers weren't in that range. This is a good general lesson: **always verify what a library function actually returns (print it, read the warning) rather than trusting the function name alone.**

**`generate_node`** — same as Phase 1's generation logic, but now shaped as a node, and (after the fix) using a slightly loosened prompt — see below.

**`out_of_scope_node`** — a trivial node that returns a fixed canned response, with **no LLM call at all**. This matters for cost/latency: if we can cheaply determine a question is unanswerable from our data, we skip an expensive/slow LLM call entirely. This is a tiny preview of why agentic routing (Plan 1 Step 2) is valuable — cheap early decisions save cost and improve UX.

**`route_after_relevance_check`** — the conditional edge function. It doesn't do any work itself; it just inspects `state["is_relevant"]` and returns a *string* naming which node to go to next. LangGraph uses that returned string to look up the next node via a mapping dict (defined in `build_graph.py`).

---

### `graph/build_graph.py` — Assembling the Graph

**Core concept: `StateGraph`, nodes, and edges.**
- `workflow.add_node("name", function)` registers a node under a string name
- `workflow.add_edge(A, B)` is a **fixed** transition: always go from A straight to B
- `workflow.add_conditional_edges(A, routing_fn, {"label1": "nodeX", "label2": "nodeY"})` is a **branching** transition: after A runs, call `routing_fn(state)`, and whatever string it returns gets looked up in the mapping to decide the real next node
- `START` and `END` are special built-in markers for the graph's entry and exit points
- `workflow.compile()` turns the declared graph into an actual runnable object, similar to how you'd compile a state machine definition into an executable one

**Why this scales well:** Adding a new branch later (e.g., a 4-way agentic router in Plan 1 Step 2) is just: add new node functions, add one more conditional edge with more mapping entries. The existing nodes (`retrieve_node`, `generate_node`) don't need to change at all. This is the entire point of doing this refactor now, even though right now it "looks like overkill" for just 2 real logic paths.

---

### `graph/debug_retrieval.py` — Diagnostic Tooling

**What it does:** A standalone script (not part of the graph itself) that runs raw `similarity_search_with_score` for a given question and prints full chunk content + distance scores, without going through the generate step.

**Why this is a genuinely important habit, not just a one-off script:** When something in a multi-step pipeline produces a wrong final answer, the instinct is often to tweak the last step (the prompt). But the actual bug can live *anywhere upstream* — in this project's case, it was in retrieval, not generation. Building a small tool to inspect *intermediate* state (what did retrieval actually return, before generation touches it) is how you correctly localize bugs in any multi-stage pipeline, ML or otherwise, instead of guessing.

---

### `graph/test_cases.py` — Regression Testing for RAG

**What it does:** A fixed batch of 10 questions (one per domain, several deliberately out-of-scope, two deliberately broad/cross-domain) run through the full graph, with expected relevance outcomes checked automatically.

**Core concept: regression testing.** As you add more phases (routing, correction loops, reranking), you risk silently breaking something that used to work. Having a small, fixed, repeatable test suite — even a simple 10-question one — lets you re-run the exact same checks after every change and immediately see if something regressed. This is the same principle as unit tests in traditional software engineering, adapted for a system whose "correctness" is fuzzier (an LLM's phrasing varies) but whose *behavioral* correctness (right domain? relevant flag correct? no hallucination?) can still be checked systematically.

---

### The Bug We Found and Fixed (worth understanding deeply)

**Symptom:** Two questions ("key findings of IPCC AR6," "economic risks of AI") were correctly flagged as relevant, correctly retrieved chunks from the right document — yet the LLM still answered "I don't have enough information."

**Root cause (found via `debug_retrieval.py`):** For broad, high-level questions, the words in the question ("key findings," "risks") appear not only in substantive content but also in **meta/structural text** — report titles, citation blocks, section headers, author lists. Those meta chunks scored as "close enough" by pure vector similarity (they do share vocabulary with the question), but they don't actually contain an answerable statement. With only `k=4` chunks retrieved, sometimes *all four* slots got filled with this kind of meta text, crowding out the real content chunks that were slightly further away in vector space.

**The fix and why it works:** Increasing `k` from 4 to 8 gives more "slots," so even if a few are wasted on meta/structural chunks, there's a much higher chance real content chunks also make it into context. Loosening the generation prompt (explicitly telling the LLM to synthesize from partial/spread-out context rather than demanding a literal, complete match) also helps the model actually *use* the good chunks that were previously being retrieved but under-utilized due to an overly strict instruction.

**Why this is explicitly *not* the final answer (important conceptual point):** This fix works by brute force — retrieve more, and ask the LLM to try harder. It doesn't fix the underlying issue, which is that **plain vector similarity search has no way to distinguish "this chunk shares vocabulary with the question" from "this chunk actually answers the question."** That distinction is exactly what a **reranker** (a second-stage model that scores query-chunk pairs for true relevance, not just embedding proximity) is for — which is Plan 1 Step 1's job. Similarly, **hybrid retrieval** (combining keyword-based BM25 search with vector search) helps because BM25 is good at exact-term matching, which can help further disambiguate "does this chunk really discuss AI's economic risks" vs. "does this chunk just contain the word risk." Both of these are queued up for later — this `k=8` fix is a legitimate, verified-working stopgap, not the end of the story.

---

## Phase 3 — Query Rewriting

### The Big Picture: Why Rewrite the Query at All?

Retrieval quality is entirely dependent on how well the *query's embedding* matches the *chunk's embedding*. If a user's raw question is vague, terse, or oddly phrased, the embedding of that raw text may not land close to the embeddings of the chunks that actually answer it — even though a well-phrased version of the same question would. Query rewriting inserts a cheap LLM call *before* retrieval to "clean up" the question into a form that's more likely to retrieve well, without changing what the user actually meant.

This is a different kind of fix than Phase 2's `k=8` band-aid: that fix widened the net *after* an imprecise query; this fix tries to make the query itself more precise *before* casting the net. Both are legitimate, complementary techniques — real RAG systems use both.

---

### `rewrite_query_node` — How It Works

**What it does:** Sends the raw question to Groq with a prompt instructing it to rewrite for clarity/specificity, preserving intent, and to return the question unchanged if it's already fine.

**Core concept: prompt-based query transformation.** This is one of the simplest members of a whole family of "query transformation" techniques in RAG (others include query decomposition — breaking a complex question into sub-questions, and HyDE — generating a hypothetical answer and embedding *that* instead of the question, which is Plan 1 Step 7 later in this project). All of them share the same underlying idea: **the text you embed for search doesn't have to be the literal text the user typed** — you can transform it first if the transformed version searches better.

**Why keep the original `question` in state alongside `rewritten_question`, rather than overwriting it?** Two reasons: (1) you want to show the user their own original question in a chat UI, not a robotically-rewritten version; (2) keeping both makes debugging much easier — if retrieval seems off, you can directly compare what was asked vs. what was actually searched for, rather than losing that information.

---

### Why the Vague-Query Test Result Matters (deep dive)

The "tell me about risks" test is worth understanding carefully because it demonstrates the **limits of what query rewriting alone can fix.**

**What rewriting *can* fix:** ambiguity in *phrasing* — typos, awkward grammar, overly terse wording, missing implied words. Given enough surrounding words in the question itself, an LLM can usually infer what's meant and produce a clearer version.

**What rewriting *cannot* fix:** ambiguity in *missing information that simply isn't in the question at all*. "Tell me about risks" doesn't contain any signal — explicit or implicit — about which domain the user means. No amount of clever rewriting can conjure information that was never provided. The LLM correctly produced a more polished *generic* version, because that's the only faithful rewrite possible without inventing an assumption the user didn't state.

**Why did the answer blend multiple domains, and is that "wrong"?** Given a generic query, our vector search (still doing plain top-8 similarity search, unchanged since Phase 2) will naturally retrieve whatever is closest across the *entire* corpus — and since "risk" is a real term discussed meaningfully in 3 of our 5 documents (AI, economics, climate), a generic query will legitimately retrieve real, relevant content from all three. The generator then does what it's instructed to do: synthesize an answer from the provided context. Every sentence in that answer is traceable to real retrieved content — so it's not hallucinated or factually wrong. It's just **not what a real user probably wanted.**

**The actual fix for this belongs to later phases, specifically:**
- **Phase 4 (conversation memory):** if this were turn 2 of a conversation where turn 1 was about AI policy, we could infer "risks" means AI risks specifically, using conversation history — no clarification needed.
- **Plan 1 Step 2 (agentic router):** could detect that a query is too ambiguous to route confidently and either ask a clarifying question back to the user, or explicitly search across domains and clearly *label* the answer as multi-domain rather than blending it seamlessly.

**The general lesson:** not every RAG problem is a retrieval problem or a generation problem — some are fundamentally an *information availability* problem (the system literally cannot know what wasn't told to it), and the correct fix is architectural (add memory, add clarification-asking) rather than tuning existing components harder.

---

## Phase 4 — Conversation Memory & Persistence

### The Big Picture: From Stateless to Stateful

Every phase so far has treated each `graph.invoke()` call as a completely independent event — no memory of anything before it. Real conversations aren't like that: people say "it," "that," "the one you mentioned," and expect the system to resolve those references using earlier context. Phase 4 turns the graph from **stateless** (each call starts from scratch) into **stateful** (each call can build on accumulated history), and makes that state **persistent** (survives even after the Python process exits and restarts).

---

### Core Concept: Checkpointing

**What a checkpointer does:** After every step (or at defined points) in a LangGraph run, the checkpointer saves a snapshot of the current state to storage, tagged with a `thread_id` (think of this like a conversation/session ID). The next time you call `graph.invoke(..., config={"configurable": {"thread_id": "same-id"}})`, LangGraph automatically loads the most recent saved state for that `thread_id` *before* running, merges your new input into it, and proceeds. This is what makes multi-turn behavior possible without you manually managing a growing history object yourself in application code.

**Why SQLite for now, not Redis/PostgreSQL (per the original roadmap)?** SQLite requires zero setup — no server process, no connection config, just a local file. It's the correct tool for *learning the checkpointing concept itself* without fighting infrastructure. Critically, LangGraph's checkpointer interface is designed so that `SqliteSaver` and something like `PostgresSaver`/`RedisSaver` are largely interchangeable — once you understand this concept, swapping the backend later (for a production-style setup, matching your original roadmap's mention of Redis/PostgreSQL) is a small, well-understood change, not a redesign.

**Why `thread_id` matters:** it's what lets a single deployed system serve *many separate, non-interfering conversations at once* — user A's conversation and user B's conversation just use different `thread_id`s, and the checkpointer keeps their histories completely separate within the same SQLite file. This is the same conceptual pattern a real multi-user chat backend would use.

---

### Two Real Bugs Hit — and Why They're Worth Understanding

**Bug 1: the closed-database error.**
`SqliteSaver.from_conn_string(...)` is a Python **generator-based context manager** — it's designed to be used as `with SqliteSaver.from_conn_string(...) as checkpointer:`, where the `with` block keeps the underlying connection alive for as long as you're inside it. Manually calling `.__enter__()` outside of an actual `with` block technically "starts" the context manager, but nothing is holding a strong reference to keep it alive — Python's garbage collector can (and did) clean it up early, silently closing the database connection out from under us. **The general lesson:** context managers exist specifically to tie a resource's lifetime to a well-defined block of code; bypassing that pattern (even when it looks like it works, like it "ran" without erroring immediately) reintroduces exactly the resource-lifetime bugs `with` blocks are designed to prevent. The fix — a plain `sqlite3.connect()` kept alive as a normal Python object for the life of the app — is simpler and more predictable specifically because it doesn't rely on generator/context-manager lifetime subtleties.

**Bug 2: history silently not persisting.**
This one didn't throw an error at all — it just quietly didn't work as intended (turn 2's rewrite stayed vague first time around). The cause: our own test code was passing `"chat_history": []` into *every* `graph.invoke()` call, including turn 2. LangGraph's checkpointer restores prior state, but any keys you *explicitly* include in your new input for that call take precedence — so we were unintentionally stomping the very history we'd just built. **The general lesson:** when using any kind of state-merging/checkpointing system, be careful about what you pass on every call versus what should be left to the system's own restoration mechanism — passing "just to be safe" defaults can silently override real accumulated state, and this class of bug is especially dangerous because it fails *quietly* (no crash, no warning) rather than loudly.

---

### Why the Multi-Hop Cross-Domain Test Result Matters (deep dive)

This is one of the most instructive results in the whole project so far, because it shows two *correctly working* components (history resolution, and grounded generation) still producing an unsatisfying end result — proving that **"every individual piece works" doesn't automatically mean "the whole system works well."**

**Step by step what happened:**
1. Turn 1 established context clearly around the WHO Health report
2. Turn 2 asked "how is that connected to climate change?" — genuinely a cross-domain bridging question
3. `rewrite_query_node` did its job correctly: it produced a fully standalone, well-formed question that explicitly named "the WHO report on global health" **and** "climate change" — a faithful, accurate rewrite of the user's intent
4. But that rewritten query, when embedded and searched, stayed vector-close to Health-doc content (because "WHO report... life expectancy... causes of death" is a lot of Health-specific vocabulary, versus one mention of "climate change") — so **all 8** retrieved chunks came from the Health document, **zero** from the Climate document, confirmed directly via `similarity_search_with_score`
5. Generation correctly refused to fabricate a connection it wasn't given evidence for — again, the grounding instruction worked exactly as designed

**So where did it actually go wrong?** Not in rewriting, not in generation — in the assumption baked into our retrieval design that **one single vector search is sufficient for any question**, even ones that inherently span two different documents/topics. A single embedding is a single point in vector space; it can be close to *Health* content or close to *Climate* content, but a genuinely 50/50 cross-domain question doesn't necessarily land near *both* — it's more likely to land near whichever domain's vocabulary happens to dominate the phrasing.

**Why this directly motivates upcoming work:**
- **Hybrid retrieval + reranking (Plan 1 Step 1)** — combining keyword-based search (BM25) with vector search, and reranking candidates from *both* Health and Climate documents, would give cross-domain content a better chance of surfacing even if the pure embedding leaned one way.
- **Agentic router with query decomposition (Plan 1 Step 2)** — the more structurally correct fix: recognizing a question like this genuinely spans two domains and **splitting it into two sub-queries** ("What does WHO say about global health?" + "What does climate change do to global health?"), retrieving separately for each, then combining — rather than forcing one query to do the work of two.

**The broader lesson to carry forward:** in agentic/RAG system design, individually-correct components chained together do not guarantee a correct end-to-end result. Testing needs to happen at the *system* level (full conversational flows, cross-domain edge cases), not just at the level of "does each node do its documented job" — because Phase 4 here is a clear example of every node doing exactly what it was designed to do, while the overall conversation still fell short of what a user would actually want.

---

*(This file will be extended with a new section after each subsequent phase completes.)*

"""
Priority 1 (Dynamic Document Ingestion) - rewritten.

Upload-specific logic ends at "produce a structured markdown Document."
From that point on, this module calls the EXACT SAME
structured_chunker.chunk_documents_structured() used for the 5 permanent
documents - no parallel chunking logic.

Pipeline:
  PDF/txt/md -> pdf_preprocessor.extract_and_clean()          [upload-only]
             -> markdown_generator.generate_structured_markdown()  [upload-only]
             -> metadata_enricher.enrich_and_persist()         [upload-only, persists to disk]
             -> structured_chunker.chunk_documents_structured()    [SHARED - same function, same code path as permanent corpus]
             -> per-thread Chroma collection (same schema/embedder as permanent KB)
             -> per-thread BM25 index (previously missing entirely - now uploads
                participate in keyword search too, not just vector search)

Session isolation preserved: each thread_id gets its own Chroma collection
AND its own BM25 index, never touching the permanent ragagent_structured
collection or the permanent BM25 index in hybrid_retriever.py. What's
shared is the CODE PATH (structured_chunker, embedder, Chroma class,
BM25Okapi), not the data.

One uploaded doc per thread_id at a time - uploading again replaces the
previous one (same policy as before).
"""
import re
import shutil
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from ingestion.embedder import get_embedding_model
from ingestion.structured_chunker import chunk_documents_structured
from ingestion import pdf_preprocessor
from ingestion import markdown_generator
from ingestion import metadata_enricher

SESSION_PERSIST_ROOT = Path(__file__).resolve().parent.parent / ".chroma_sessions"

# Cached per-thread handles so repeated turns in the same conversation don't
# reopen Chroma from disk or rebuild BM25 every retrieval call.
_session_vectorstore_cache: dict = {}
_session_bm25_cache: dict = {}  # thread_id -> (BM25Okapi, list[Document])


def _collection_name(thread_id: str) -> str:
    return f"session_{thread_id}"


def _persist_dir(thread_id: str) -> str:
    return str(SESSION_PERSIST_ROOT / thread_id)


def _tokenize(text: str) -> list[str]:
    """Same tokenizer as hybrid_retriever.py, kept identical so fused BM25
    scores are computed consistently regardless of which index they came from."""
    return re.findall(r"\b\w+\b", text.lower())


def build_session_vectorstore(thread_id: str, file_path: Path, original_filename: str) -> int:
    """
    Full Priority-1 ingestion pipeline for one uploaded file. Returns the
    number of chunks stored. Replaces any prior upload for this thread_id.
    """
    # Stage 1+2: Extract + Clean (upload-only)
    cleaned_text = pdf_preprocessor.extract_and_clean(file_path)
    if not cleaned_text.strip():
        raise ValueError(f"No extractable text found in '{original_filename}'")

    # Stage 3: Markdown Generator (upload-only) - structure, never rewrite content
    structured_body = markdown_generator.generate_structured_markdown(cleaned_text)

    # Stage 4: Metadata Generation/Enrichment + Persist (upload-only)
    markdown_path = metadata_enricher.enrich_and_persist(structured_body, original_filename, thread_id)

    # From here on: IDENTICAL code path to the permanent corpus.
    full_markdown = markdown_path.read_text(encoding="utf-8")
    doc = Document(page_content=full_markdown, metadata={"source": original_filename})
    chunks = chunk_documents_structured([doc])  # SAME function used for the 5 permanent docs

    if not chunks:
        raise ValueError(f"Structuring/chunking produced no chunks for '{original_filename}'")

    delete_session_vectorstore(thread_id)  # replace, don't accumulate

    # Same embedder, same Chroma class as the permanent KB - separate collection only.
    embedder = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        collection_name=_collection_name(thread_id),
        persist_directory=_persist_dir(thread_id),
    )
    _session_vectorstore_cache[thread_id] = vectorstore

    # Per-thread BM25 index - same BM25Okapi/tokenizer as the permanent index,
    # just scoped to this thread's chunks instead of the whole corpus.
    tokenized = [_tokenize(c.page_content) for c in chunks]
    _session_bm25_cache[thread_id] = (BM25Okapi(tokenized), chunks)

    return len(chunks)


def has_session_doc(thread_id: str) -> bool:
    """Cheap existence check - does this thread have an uploaded doc?"""
    if not thread_id:
        return False
    if thread_id in _session_vectorstore_cache:
        return True
    return Path(_persist_dir(thread_id)).exists()


def get_session_vectorstore(thread_id: str):
    """Returns the Chroma vectorstore for this thread's uploaded doc, or None."""
    if not has_session_doc(thread_id):
        return None
    if thread_id not in _session_vectorstore_cache:
        embedder = get_embedding_model()
        _session_vectorstore_cache[thread_id] = Chroma(
            collection_name=_collection_name(thread_id),
            embedding_function=embedder,
            persist_directory=_persist_dir(thread_id),
        )
    return _session_vectorstore_cache[thread_id]


def get_session_bm25(thread_id: str):
    """
    Returns (BM25Okapi, chunks) for this thread's uploaded doc, or (None, [])
    if none exists or it hasn't been rebuilt this process (e.g. after a
    restart with persisted Chroma but no in-memory BM25 cache - BM25 isn't
    persisted to disk since it's cheap to rebuild from the vectorstore's
    stored chunks). Used by hybrid_retriever.py to fold session chunks into
    the SAME fusion+rerank pipeline as the permanent corpus - never a
    separate retrieval path.
    """
    if not has_session_doc(thread_id):
        return None, []
    if thread_id in _session_bm25_cache:
        return _session_bm25_cache[thread_id]

    # Process restarted / cache cold: rebuild BM25 from the persisted Chroma collection's chunks.
    vectorstore = get_session_vectorstore(thread_id)
    stored = vectorstore.get()  # {"documents": [...], "metadatas": [...]}
    chunks = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(stored["documents"], stored["metadatas"])
    ]
    if not chunks:
        return None, []
    tokenized = [_tokenize(c.page_content) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    _session_bm25_cache[thread_id] = (bm25, chunks)
    return bm25, chunks


def delete_session_vectorstore(thread_id: str) -> None:
    """Deletes a thread's uploaded doc, its Chroma collection, its BM25 cache,
    and its persisted markdown file from disk. Safe to call even if nothing
    exists for this thread_id."""
    _session_vectorstore_cache.pop(thread_id, None)
    _session_bm25_cache.pop(thread_id, None)

    persist_path = Path(_persist_dir(thread_id))
    if persist_path.exists():
        shutil.rmtree(persist_path, ignore_errors=True)

    markdown_path = metadata_enricher.UPLOADED_MARKDOWN_DIR / f"{thread_id}.md"
    markdown_path.unlink(missing_ok=True)

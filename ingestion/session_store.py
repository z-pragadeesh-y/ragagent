"""
Plan 3: Session-scoped document store for user-uploaded files.

Uploaded docs are NEVER merged into the permanent knowledge base
(ingestion/vectorstore.py's ragagent_structured collection) - each upload
gets its own isolated Chroma collection keyed by thread_id, so one user's
upload can never leak into another user's retrieval or pollute the
permanent 5-domain KB. Reuses the same embedding model (ingestion/embedder.py)
as the main pipeline for consistency.

Supports .pdf, .txt, and .md uploads. PDF extraction uses pypdf.

One uploaded doc per thread_id at a time - uploading again to the same
thread_id replaces the previous one, rather than silently accumulating,
to keep retrieval-merging in graph/nodes.py simple and predictable.
"""
import shutil
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.embedder import get_embedding_model

SESSION_PERSIST_ROOT = Path(__file__).resolve().parent.parent / ".chroma_sessions"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Cached vectorstore handles per thread_id, so repeated retrievals in the
# same conversation don't reopen the Chroma collection from disk every turn.
_session_vectorstore_cache: dict = {}


def _collection_name(thread_id: str) -> str:
    return f"session_{thread_id}"


def _persist_dir(thread_id: str) -> str:
    return str(SESSION_PERSIST_ROOT / thread_id)


def _extract_text(file_path: Path) -> str:
    """Extracts raw text from an uploaded file based on its extension."""
    suffix = file_path.suffix.lower()

    if suffix in (".txt", ".md"):
        return file_path.read_text(encoding="utf-8")

    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .txt, .md")


def build_session_vectorstore(thread_id: str, file_path: Path, original_filename: str) -> int:
    """
    Extracts, chunks, and embeds an uploaded document into a Chroma collection
    scoped ONLY to this thread_id. Returns the number of chunks stored.
    Replaces any prior upload for this thread_id first.
    """
    text = _extract_text(file_path)
    if not text.strip():
        raise ValueError(f"No extractable text found in '{original_filename}'")

    doc = Document(
        page_content=text,
        metadata={
            "source": original_filename,
            "domain_tag": "uploaded",
            "document_title": original_filename,
            "section_title": "",
        },
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len,
    )
    chunks = splitter.split_documents([doc])
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    delete_session_vectorstore(thread_id)  # replace, don't accumulate

    embedder = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        collection_name=_collection_name(thread_id),
        persist_directory=_persist_dir(thread_id),
    )
    _session_vectorstore_cache[thread_id] = vectorstore

    return len(chunks)


def has_session_doc(thread_id: str) -> bool:
    """Cheap existence check - does this thread have an uploaded doc?"""
    if not thread_id:
        return False
    if thread_id in _session_vectorstore_cache:
        return True
    return Path(_persist_dir(thread_id)).exists()


def load_session_vectorstore(thread_id: str):
    """Returns the Chroma vectorstore for this thread's uploaded doc, or None if none exists."""
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


def session_retrieve(thread_id: str, query: str, k: int = 4) -> list[Document]:
    """Similarity search against this thread's uploaded doc only. Returns [] if none exists."""
    vectorstore = load_session_vectorstore(thread_id)
    if vectorstore is None:
        return []
    return vectorstore.similarity_search(query, k=k)


def delete_session_vectorstore(thread_id: str) -> None:
    """Deletes a thread's uploaded doc + its Chroma collection from disk. Safe to call even
    if nothing exists for this thread_id. Call this when a session/thread ends (or via the
    DELETE /upload/{thread_id} endpoint) so uploads don't accumulate indefinitely on disk -
    note Railway's filesystem is ephemeral anyway and wipes on every redeploy, but this
    matters for long-running deployments between redeploys."""
    _session_vectorstore_cache.pop(thread_id, None)
    persist_path = Path(_persist_dir(thread_id))
    if persist_path.exists():
        shutil.rmtree(persist_path, ignore_errors=True)

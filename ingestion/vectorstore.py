"""
Builds (or loads) a persistent Chroma vector store from document chunks.
"""
from pathlib import Path
from langchain_chroma import Chroma
from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents
from ingestion.embedder import get_embedding_model

PERSIST_DIR = str(Path(__file__).resolve().parent.parent / ".chroma")
COLLECTION_NAME = "ragagent_phase1"

def build_vectorstore():
    """Embeds all chunks and persists them to a local Chroma DB."""
    docs = load_documents()
    chunks = chunk_documents(docs)
    embedder = get_embedding_model()

    print(f"Embedding and storing {len(chunks)} chunks... (this takes a minute or two on CPU)")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )

    print(f"Vector store built and persisted to: {PERSIST_DIR}")
    return vectorstore


_cached_vectorstore = None

def load_vectorstore():
    """Loads an already-built Chroma DB from disk (cached after first call)."""
    global _cached_vectorstore
    if _cached_vectorstore is None:
        embedder = get_embedding_model()
        _cached_vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedder,
            persist_directory=PERSIST_DIR,
        )
    return _cached_vectorstore

if __name__ == "__main__":
    vs = build_vectorstore()

    print(f"\nTotal vectors stored: {vs._collection.count()}")

    # Quick sanity search
    query = "What is climate change causing globally?"
    results = vs.similarity_search(query, k=3)

    print(f"\n--- Top 3 results for test query: '{query}' ---")
    for i, r in enumerate(results):
        print(f"\n[{i+1}] Source: {r.metadata.get('source')}")
        print(r.page_content[:200])
STRUCTURED_COLLECTION_NAME = "ragagent_structured"

def build_structured_vectorstore():
    """Embeds structured chunks (with domain_tag/section metadata) into a separate Chroma collection."""
    from ingestion.structured_chunker import chunk_documents_structured

    docs = load_documents()
    chunks = chunk_documents_structured(docs)
    embedder = get_embedding_model()

    print(f"Embedding and storing {len(chunks)} structured chunks...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        collection_name=STRUCTURED_COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )

    print(f"Structured vector store built and persisted to: {PERSIST_DIR}")
    return vectorstore


_cached_structured_vectorstore = None

def load_structured_vectorstore():
    """Loads the structured Chroma collection (cached after first call)."""
    global _cached_structured_vectorstore
    if _cached_structured_vectorstore is None:
        embedder = get_embedding_model()
        _cached_structured_vectorstore = Chroma(
            collection_name=STRUCTURED_COLLECTION_NAME,
            embedding_function=embedder,
            persist_directory=PERSIST_DIR,
        )
    return _cached_structured_vectorstore
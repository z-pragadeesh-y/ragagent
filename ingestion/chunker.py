"""
Splits loaded documents into fixed-size overlapping chunks.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from ingestion.loader import load_documents

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def chunk_documents(documents: list[Document]) -> list[Document]:
    """Splits documents into overlapping fixed-size chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # Add a chunk_id to metadata for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)

    print(f"Total chunks created: {len(chunks)}\n")

    # Per-source breakdown
    from collections import Counter
    counts = Counter(c.metadata["source"] for c in chunks)
    for source, count in counts.items():
        print(f"- {source}: {count} chunks")

    print("\n--- Sample chunk (#0) ---")
    print(chunks[0].page_content[:300])
    print("...")
    print("\nMetadata:", chunks[0].metadata)
"""
Loads all markdown documents from data/raw/ into LangChain Document objects.
"""
import os
from pathlib import Path
from langchain_core.documents import Document

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

def load_documents() -> list[Document]:
    """Reads every .md file in data/raw/ and returns a list of Documents."""
    documents = []

    for file_path in sorted(RAW_DATA_DIR.glob("*.md")):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        doc = Document(
            page_content=content,
            metadata={"source": file_path.name}
        )
        documents.append(doc)

    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents\n")
    for d in docs:
        print(f"- {d.metadata['source']}: {len(d.page_content)} characters")
"""
Structure-aware chunking (Plan 2, Step 1): splits documents on their actual
section boundaries (using the section_title/parent_section frontmatter blocks
already present in the source files) instead of blind fixed-size splitting,
and attaches domain_tag + section metadata to every chunk. This is the
metadata Phase 1's chunker has ignored since day one.
"""
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.loader import load_documents

MAX_SECTION_CHUNK_SIZE = 1200  # sections longer than this still get sub-split
CHUNK_OVERLAP = 200

DOC_FRONTMATTER_PATTERN = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
SECTION_MARKER_PATTERN = re.compile(
    r'---\n(?:section_number:\s*.*?\n)?section_title:\s*(.*?)\nparent_section:\s*(.*?)\n---\n',
    re.DOTALL
)


def _parse_doc_frontmatter(content: str) -> dict:
    """Extracts domain_tag and document_title from the top-level YAML block."""
    match = DOC_FRONTMATTER_PATTERN.match(content)
    if not match:
        return {}
    block = match.group(1)
    fields = {}
    for line in block.split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields


def _split_into_sections(content: str) -> list[dict]:
    """Splits content on section-marker blocks, returning [{section_title, parent_section, text}]."""
    matches = list(SECTION_MARKER_PATTERN.finditer(content))
    sections = []

    if not matches:
        # No section markers found at all (shouldn't happen given our corpus, but fail safe)
        return [{"section_title": "", "parent_section": "", "text": content}]

    for i, match in enumerate(matches):
        section_title = match.group(1).strip()
        parent_section = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        text = content[start:end].strip()
        if text:
            sections.append({
                "section_title": section_title,
                "parent_section": parent_section,
                "text": text,
            })
    return sections


def chunk_documents_structured(documents: list[Document]) -> list[Document]:
    """Structure-aware chunking: splits on section boundaries, attaches domain_tag +
    section metadata to every chunk. Sections longer than MAX_SECTION_CHUNK_SIZE are
    further sub-split with overlap, same as the baseline chunker."""
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_SECTION_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    all_chunks = []
    chunk_id = 0

    for doc in documents:
        frontmatter = _parse_doc_frontmatter(doc.page_content)
        domain_tag = frontmatter.get("domain_tag", "unknown")
        document_title = frontmatter.get("document_title", "")

        sections = _split_into_sections(doc.page_content)

        for section in sections:
            base_metadata = {
                "source": doc.metadata["source"],
                "domain_tag": domain_tag,
                "document_title": document_title,
                "section_title": section["section_title"],
                "parent_section": section["parent_section"],
            }

            if len(section["text"]) <= MAX_SECTION_CHUNK_SIZE:
                all_chunks.append(Document(
                    page_content=section["text"],
                    metadata={**base_metadata, "chunk_id": chunk_id},
                ))
                chunk_id += 1
            else:
                sub_chunks = sub_splitter.split_text(section["text"])
                for sub_text in sub_chunks:
                    all_chunks.append(Document(
                        page_content=sub_text,
                        metadata={**base_metadata, "chunk_id": chunk_id},
                    ))
                    chunk_id += 1

    return all_chunks


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents_structured(docs)

    print(f"Total structured chunks created: {len(chunks)}\n")

    from collections import Counter
    domain_counts = Counter(c.metadata["domain_tag"] for c in chunks)
    for domain, count in domain_counts.items():
        print(f"- {domain}: {count} chunks")

    print("\n--- Sample chunk (#5) ---")
    sample = chunks[5]
    print("Metadata:", sample.metadata)
    print("Content:", sample.page_content[:300])
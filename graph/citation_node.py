"""
Plan 2 Step 3: Citation & Attribution node.

Sits immediately after `generate`, before the answer is returned (per the
roadmap's stated architecture) — kept separate from generate_node so
generation and citation-formatting stay independent responsibilities.

Builds the References list from REAL chunk metadata (source, section,
domain_tag) — never invented by the LLM — and validates that every
[Source N] the LLM cited inline actually corresponds to a real retrieved
chunk, stripping any that don't (a model citing [Source 7] when only 4
chunks were retrieved is a hallucinated citation, and silently leaving it
in the answer would defeat the entire point of this step: verifiability).
"""
import re
import logging

logger = logging.getLogger("llm_manager")

CITATION_PATTERN = re.compile(r"\[Source (\d+)\]")


def _build_references(docs) -> tuple[list[dict], str]:
    """
    Builds structured citation records (for API consumers / logging) and a
    matching human-readable References block, both purely from real chunk
    metadata attached during Plan 2 Step 1 (structure-aware chunking).
    Returns (citations, references_text).
    """
    citations = []
    reference_lines = []

    for i, doc in enumerate(docs, start=1):
        source_file = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section_title", "")
        doc_title = doc.metadata.get("document_title", "")
        domain_tag = doc.metadata.get("domain_tag", "")

        citations.append({
            "index": i,
            "source": source_file,
            "document_title": doc_title,
            "section_title": section,
            "domain_tag": domain_tag,
        })

        label_bits = [b for b in [doc_title or source_file, section] if b]
        label = " — ".join(label_bits)
        if domain_tag:
            label = f"{label} [{domain_tag}]" if label else f"[{domain_tag}]"
        reference_lines.append(f"[Source {i}] {label} ({source_file})")

    references_text = "\n".join(reference_lines)
    return citations, references_text


def _strip_invalid_citations(answer: str, valid_count: int) -> tuple[str, list[int]]:
    """
    Finds every [Source N] marker in the answer; any N outside 1..valid_count
    is a hallucinated citation (references a chunk that was never retrieved).
    Removes those specific markers from the text and returns which invalid
    numbers were found, for logging.
    """
    invalid_found = []

    def _check(match):
        n = int(match.group(1))
        if n < 1 or n > valid_count:
            invalid_found.append(n)
            return ""  # drop the hallucinated marker entirely
        return match.group(0)

    cleaned = CITATION_PATTERN.sub(_check, answer)
    return cleaned, invalid_found


def citation_node(state: dict) -> dict:
    """
    Validates inline [Source N] citations against the real retrieved_docs,
    strips any hallucinated ones, and appends a References list built from
    real chunk metadata. Also returns `citations` as structured data on
    state, so API consumers (main.py) can surface citations separately from
    the answer text if desired, rather than only as appended plain text.
    """
    docs = state.get("retrieved_docs", [])
    raw_answer = state.get("answer", "")

    if not docs:
        # direct_answer / out_of_scope paths reach here with no retrieved_docs
        # if wired in unconditionally — nothing to cite, pass through untouched.
        return {"answer": raw_answer, "citations": []}

    citations, references_text = _build_references(docs)
    cleaned_answer, invalid_numbers = _strip_invalid_citations(raw_answer, len(docs))

    if invalid_numbers:
        logger.warning(
            "citation_node: stripped %d hallucinated citation marker(s) %s — "
            "only %d source(s) were actually retrieved",
            len(invalid_numbers), invalid_numbers, len(docs),
        )

    final_answer = cleaned_answer
    if references_text:
        final_answer = f"{cleaned_answer}\n\nReferences:\n{references_text}"

    return {"answer": final_answer, "citations": citations}

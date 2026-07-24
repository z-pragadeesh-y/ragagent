"""
Priority 1 (Dynamic Document Ingestion) - Stage 1 & 2: Extract + Clean.

Pure text-in/text-out. No structuring, no LLM, no metadata. This module's
only job is to take a raw uploaded file and produce clean plain text -
equivalent to what a human would have in hand before manually writing one
of the 5 original structured markdown documents. Everything downstream
(markdown_generator.py, metadata_enricher.py, structured_chunker.py) is
then IDENTICAL for uploaded and permanent documents.
"""
import re
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

# Matches a line that is ONLY a page number (possibly with surrounding
# whitespace/dashes), e.g. "12", "- 12 -", "Page 12", "Page 12 of 40".
_PAGE_NUMBER_LINE = re.compile(
    r"^\s*(?:-\s*)?(?:page\s*)?\d{1,4}(?:\s*of\s*\d{1,4})?(?:\s*-)?\s*$",
    re.IGNORECASE,
)


def extract_raw_text(file_path: Path) -> str:
    """Extracts raw, unprocessed text from an uploaded file based on extension."""
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    if suffix in (".txt", ".md"):
        return file_path.read_text(encoding="utf-8")

    # .pdf
    from pypdf import PdfReader
    reader = PdfReader(str(file_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _find_repeated_lines(pages: list[str], min_repeats: int = 3) -> set[str]:
    """Identifies lines (e.g. running headers/footers) that repeat verbatim
    across many pages - a strong signal they're boilerplate, not content."""
    from collections import Counter
    counts = Counter()
    for page_text in pages:
        seen_in_page = set(line.strip() for line in page_text.split("\n") if line.strip())
        counts.update(seen_in_page)
    return {line for line, n in counts.items() if n >= min_repeats and len(line) < 120}


def clean_text(raw_text: str, page_texts: list[str] | None = None) -> str:
    """
    Removes headers/footers, page-number-only lines, and normalizes whitespace.
    Does NOT touch actual sentence content - conservative by design, since
    over-aggressive cleaning risks silently dropping real information.

    page_texts, if provided (per-page extraction), lets us detect repeated
    running headers/footers across pages before they're joined into one blob.
    """
    repeated_lines = _find_repeated_lines(page_texts) if page_texts else set()

    lines = raw_text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")  # preserve paragraph breaks
            continue
        if stripped in repeated_lines:
            continue
        if _PAGE_NUMBER_LINE.match(stripped):
            continue
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)

    # Collapse 3+ blank lines down to 2 (paragraph break), collapse runs of spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_and_clean(file_path: Path) -> str:
    """Full Stage 1+2 pipeline: extract, then clean. Convenience wrapper."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        page_texts = [page.extract_text() or "" for page in reader.pages]
        raw_text = "\n\n".join(page_texts)
        return clean_text(raw_text, page_texts=page_texts)

    raw_text = extract_raw_text(file_path)
    return clean_text(raw_text)

"""
Priority 1 - Stage 3: Markdown Generator.

Responsibility: ONLY reformat cleaned text into the same structured markdown
shape used by the 5 permanent corpus documents (section markers matching
structured_chunker.py's SECTION_MARKER_PATTERN). This is a formatting/
structuring stage - it must NOT summarize, compress, rewrite, or reinterpret
the source content. The LLM is instructed to organize and label, not author.

Does NOT generate document-level metadata (domain, domain_tag, title) -
that is metadata_enricher.py's separate responsibility, kept modular per
the approved architecture.

Output is validated against structured_chunker's own section-marker regex
before being accepted; if the LLM fails to produce valid markers, the
document falls back to a single unnamed section rather than crashing.
"""
import re
import logging
from langchain_core.prompts import ChatPromptTemplate

from llm.task_router import get_llm
from llm.errors import AllProvidersFailedError

logger = logging.getLogger("llm_manager")

# Reuses the exact pattern structured_chunker.py parses, so generated output
# is guaranteed compatible with the shared chunker - no format drift possible.
SECTION_MARKER_PATTERN = re.compile(
    r'---\n(?:section_number:\s*.*?\n)?section_title:\s*(.*?)\nparent_section:\s*(.*?)\n---\n',
    re.DOTALL
)

MARKDOWN_GENERATION_PROMPT = """You are a document structuring tool. Your ONLY job is to organize the
following cleaned document text into sections with clear headings. You must NOT summarize, shorten,
rewrite, paraphrase, or omit any content - every sentence of the original text must appear in your
output, verbatim, just organized under section headings.

For each natural section you identify (based on topic shifts, existing headings in the text, or logical
divisions), output it in exactly this format:

### <Section Heading>

---
section_title: <Section Heading>
parent_section: <a broader parent heading, or "Document" if this is a top-level section>
---

<the FULL original text belonging to this section, completely unchanged>

Rules:
- Preserve every word of the original content exactly - this is a structuring task, not a summarization task.
- If the document has no obvious internal structure, use a single section titled "Full Document" with
  parent_section "Document" and place ALL the text inside it.
- Do not add commentary, analysis, or any text that wasn't in the original document.
- Do not invent facts, headings, or content not present in the source.

Document text:
{text}

Structured output:"""


def _validate_sections(markdown: str) -> bool:
    """Checks the LLM output actually contains parseable section markers matching
    structured_chunker.py's regex - our compatibility contract with the shared chunker."""
    return len(SECTION_MARKER_PATTERN.findall(markdown)) > 0


def _fallback_single_section(cleaned_text: str) -> str:
    """No-LLM-needed fallback: wraps the whole cleaned text as one section,
    guaranteeing structured_chunker.py can still parse it."""
    return (
        "### Full Document\n\n"
        "---\n"
        "section_title: Full Document\n"
        "parent_section: Document\n"
        "---\n\n"
        f"{cleaned_text}\n"
    )


WINDOW_SIZE = 3000  # chars per LLM call - large enough for real section context,
                     # small enough that the LLM can safely re-emit the FULL window
                     # verbatim within its output token limit (this is what caused
                     # silent truncation -> single-section fallback on large docs)
WINDOW_OVERLAP = 0   # no overlap needed: we split on paragraph boundaries below,
                     # so windows don't cut mid-sentence, and duplicated content
                     # from overlap would corrupt chunk counts/citations


def _split_into_windows(cleaned_text: str, window_size: int = WINDOW_SIZE) -> list[str]:
    """Splits cleaned text into windows on paragraph boundaries (blank lines),
    never mid-sentence, each close to but not exceeding window_size chars."""
    paragraphs = cleaned_text.split("\n\n")
    windows = []
    current = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # account for the "\n\n" separator
        if current and current_len + para_len > window_size:
            windows.append("\n\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        windows.append("\n\n".join(current))

    return windows


def _generate_single_window(window_text: str, window_index: int) -> str:
    """Runs one window through the LLM. Falls back to a single-section wrapper
    for JUST this window on failure/unparseable output - a bad window never
    takes down the whole document's structuring."""
    llm = get_llm(task="generate", temperature=0)
    prompt = ChatPromptTemplate.from_template(MARKDOWN_GENERATION_PROMPT)
    chain = prompt | llm

    try:
        response = chain.invoke({"text": window_text})
        structured = response.content.strip()
    except AllProvidersFailedError:
        logger.warning("markdown_generator: window %d - all providers unavailable, using single-section fallback for this window", window_index)
        return _fallback_single_section(window_text)

    if not _validate_sections(structured):
        logger.warning("markdown_generator: window %d - LLM output had no valid section markers, using single-section fallback for this window", window_index)
        return _fallback_single_section(window_text)

    return structured


def generate_structured_markdown(cleaned_text: str) -> str:
    """
    Converts cleaned plain text into structured markdown body (section markers
    only - NOT the document-level frontmatter, which metadata_enricher.py adds
    separately). Large documents are processed in windows so a single LLM call
    never has to re-emit an entire document's content verbatim (which risks
    silent truncation and a fallback to one flattened section) - each window
    is structured independently and the results concatenated, preserving real
    section-level granularity throughout the whole document.
    """
    windows = _split_into_windows(cleaned_text)
    structured_parts = [_generate_single_window(w, i) for i, w in enumerate(windows)]
    return "\n\n".join(structured_parts)

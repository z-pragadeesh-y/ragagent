"""
Priority 1 - Stage 4: Metadata Generation / Enrichment.

Kept separate from markdown_generator.py per the approved architecture -
one module, one responsibility. Takes a section-structured markdown body
(from markdown_generator.py) and prepends the document-level YAML
frontmatter block that structured_chunker.py's _parse_doc_frontmatter()
reads (document_title, domain, domain_tag, source, document_type, etc.) -
the exact same frontmatter shape used by the 5 permanent documents.

Domain classification: does NOT force every upload into one of the 5
existing domains. If the LLM confidently matches one of the 5, use it;
otherwise assign domain="uploaded", domain_tag="uploaded", keeping the
system future-proof for arbitrary document types (per approved architecture
point 4).

Also handles Stage: Persist - writes the final, complete structured
markdown (frontmatter + body) to disk before chunking, so uploads are
inspectable/re-indexable exactly like the permanent corpus's .md files.
"""
import json
import logging
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate

from llm.task_router import get_llm
from llm.errors import AllProvidersFailedError

logger = logging.getLogger("llm_manager")

KNOWN_DOMAINS = {
    "AI policy/risk management": "ai_policy",
    "climate change": "climate",
    "global economics": "economics",
    "public health": "health",
    "AI/NLP research": "ai_research",
}

METADATA_PROMPT = """You are classifying a document's title and domain for a knowledge base that has
5 existing domains:
- AI policy/risk management (domain_tag: ai_policy)
- climate change (domain_tag: climate)
- global economics (domain_tag: economics)
- public health (domain_tag: health)
- AI/NLP research (domain_tag: ai_research)

Read the document excerpt below. Respond with ONLY a JSON object, no other text:
{{"document_title": "<a concise title for this document>", "matches_existing_domain": true or false,
"domain_tag": "<one of: ai_policy, climate, economics, health, ai_research> or 'uploaded' if it does not
clearly match any of the 5", "domain": "<human-readable domain name, or 'Uploaded Document' if none match>"}}

Only set matches_existing_domain to true if the document is CLEARLY and primarily about one of the 5
domains above - do not force a weak or partial match.

Document excerpt (may be truncated):
{excerpt}

JSON response:"""

# Session-scoped uploads are persisted here, mirroring data/raw/ for the
# permanent corpus but kept physically separate (never mixed with it).
UPLOADED_MARKDOWN_DIR = Path(__file__).resolve().parent.parent / "data" / "uploaded_markdown"


def _classify_metadata(cleaned_excerpt: str, original_filename: str) -> dict:
    """LLM call to infer document_title + domain classification. Falls back to
    safe 'uploaded' defaults on any failure - never blocks the upload."""
    llm = get_llm(task="grade", temperature=0)  # structured classification, SIMPLE lane
    prompt = ChatPromptTemplate.from_template(METADATA_PROMPT)
    chain = prompt | llm

    fallback = {
        "document_title": original_filename,
        "domain": "Uploaded Document",
        "domain_tag": "uploaded",
    }

    try:
        response = chain.invoke({"excerpt": cleaned_excerpt[:3000]})
        raw = response.content.strip()
    except AllProvidersFailedError:
        logger.warning("metadata_enricher: LLM unavailable, defaulting to domain_tag=uploaded")
        return fallback

    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()

    try:
        parsed = json.loads(raw)
        if not parsed.get("matches_existing_domain") or parsed.get("domain_tag") not in KNOWN_DOMAINS.values():
            return {
                "document_title": parsed.get("document_title", original_filename),
                "domain": "Uploaded Document",
                "domain_tag": "uploaded",
            }
        return {
            "document_title": parsed.get("document_title", original_filename),
            "domain": parsed.get("domain", "Uploaded Document"),
            "domain_tag": parsed["domain_tag"],
        }
    except (json.JSONDecodeError, KeyError):
        logger.warning("metadata_enricher: could not parse classification JSON, defaulting to domain_tag=uploaded")
        return fallback


def _build_frontmatter(metadata: dict, original_filename: str) -> str:
    """Builds the document-level YAML frontmatter block, same shape structured_chunker.py parses."""
    return (
        "---\n"
        f"document_title: {metadata['document_title']}\n"
        f"domain: {metadata['domain']}\n"
        f"domain_tag: {metadata['domain_tag']}\n"
        f"source: {original_filename}\n"
        "document_type: user-uploaded document\n"
        "---\n\n"
    )


def enrich_and_persist(structured_body: str, original_filename: str, thread_id: str) -> Path:
    """
    Classifies domain/title, prepends frontmatter, and persists the complete
    structured markdown document to disk (Stage: Persist). Returns the path
    to the saved file, which loader-equivalent code then reads exactly like
    a permanent-corpus .md file.
    """
    metadata = _classify_metadata(structured_body, original_filename)
    frontmatter = _build_frontmatter(metadata, original_filename)
    full_document = frontmatter + structured_body

    UPLOADED_MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = UPLOADED_MARKDOWN_DIR / f"{thread_id}.md"
    out_path.write_text(full_document, encoding="utf-8")

    logger.info(
        "metadata_enricher: persisted structured markdown for thread %s -> %s (domain_tag=%s)",
        thread_id, out_path, metadata["domain_tag"],
    )
    return out_path

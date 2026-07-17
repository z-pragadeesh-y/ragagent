"""
HyDE (Hypothetical Document Embeddings): instead of embedding the raw query,
generate a short hypothetical answer passage and embed THAT for vector search.
The hypothetical passage tends to be closer, in embedding space, to real
answer passages than the bare question is - directly targets query/answer
embedding mismatch (e.g. short questions vs. long descriptive document text).
"""
from langchain_core.prompts import ChatPromptTemplate
from llm.task_router import get_llm
from llm.errors import AllProvidersFailedError

HYDE_PROMPT_TEMPLATE = """Write a short, factual passage (2-4 sentences) that would plausibly answer
the following question, in the style of a technical report or policy document excerpt.

Question: {question}

CRITICAL: Output ONLY the passage itself. Do not include any preamble like "Here is a passage...",
any heading, any title, any notes, disclaimers, or caveats about the passage. Do not mention that
this is hypothetical or that it doesn't cite a real source. Just write the passage text directly,
starting immediately with the first sentence of content.

Passage:"""


def generate_hyde_passage(question: str) -> str:
    """Returns a hypothetical answer passage, or the original question on failure."""
    llm = get_llm(task="hyde", temperature=0.2)
    prompt = ChatPromptTemplate.from_template(HYDE_PROMPT_TEMPLATE)
    chain = prompt | llm

    try:
        response = chain.invoke({"question": question})
        passage = response.content.strip()
    except AllProvidersFailedError:
        return question  # fail safe: fall back to raw query if HyDE generation fails

    # Defensive cleanup: strip common preamble/meta-commentary patterns even if
    # the model doesn't perfectly follow the "output only the passage" instruction
    lines = passage.split("\n")
    cleaned_lines = []
    skip_prefixes = ("here is", "here's", "note:", "**note", "this passage", "excerpt from")
    for line in lines:
        stripped = line.strip().lower()
        if any(stripped.startswith(p) for p in skip_prefixes):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned if cleaned else passage  # never return empty string
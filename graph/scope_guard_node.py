"""
Plan 2 Step 9 (roadmap) / Plan 2 Step 4 (local numbering): Output guardrail —
scope enforcement.

Runs after citation, before update_history. Checks whether the final,
citation-annotated answer stayed within the knowledge base's factual scope
(AI policy, climate, economics, public health, AI research) or drifted into
opinions/advice/recommendations the retrieved context doesn't actually
support. LLM-based (semantic judgment, not something regex can catch),
routed through the SIMPLE lane (task="grade") — same reasoning as
injection_guard_node's fallback: structured classification, not generation.

If flagged, the answer (and any citations built from it) is discarded and
replaced with a fixed scope-refusal message, rather than silently letting
scope-violating content reach the user with a References list attached that
would misleadingly imply the drifted content was itself source-grounded.

Plan 2 Step 5 update: also sets scope_flagged on state (mirroring
is_injection's pattern) so feedback logging can capture whether this
guardrail fired, without needing to parse log text — detection logic itself
is unchanged from Step 4.
"""
import logging
from langchain_core.prompts import ChatPromptTemplate

from graph.state import RAGState
from llm.task_router import get_llm
from llm.errors import AllProvidersFailedError

logger = logging.getLogger("llm_manager")

SCOPE_VIOLATION_MESSAGE = (
    "I can only provide factual information grounded in the knowledge base "
    "(AI policy, climate, economics, public health, AI research) — not "
    "personal opinions or advice beyond that. Could you rephrase your "
    "question to ask about factual content in one of those areas?"
)

SCOPE_CHECK_PROMPT = """You are reviewing an AI assistant's answer for scope compliance. The assistant is
only supposed to provide factual information grounded in a knowledge base covering AI policy, climate
science, economics, public health, and AI research — never personal opinions, subjective
recommendations, or advice that goes beyond what the source material factually supports.

Question: {question}

Answer: {answer}

Does this answer stay within factual scope, or does it drift into opinion/advice/recommendations not
grounded in the knowledge base? Respond with ONLY "yes" (drifts out of scope) or "no" (stays in scope).

Answer (yes/no):"""


def _llm_flags_scope_violation(question: str, answer: str) -> bool:
    """SIMPLE lane classification. Fails open (assumes in-scope) if all
    providers are down, so a provider outage doesn't take down every
    retrieval-based answer — same fail-open reasoning as injection_guard_node."""
    llm = get_llm(task="grade", temperature=0)
    prompt = ChatPromptTemplate.from_template(SCOPE_CHECK_PROMPT)
    chain = prompt | llm

    try:
        response = chain.invoke({"question": question, "answer": answer})
        verdict = response.content.strip().lower()
    except AllProvidersFailedError:
        logger.warning("scope_guard_node: LLM check unavailable, failing open (treating as in-scope)")
        return False

    return verdict.startswith("yes")


def scope_guard_node(state: RAGState) -> dict:
    """Checks the final answer for scope drift. If flagged, discards the
    answer and its citations, replacing both with a fixed refusal and setting
    scope_flagged=True — otherwise passes the answer through unchanged with
    scope_flagged=False."""
    answer = state.get("answer", "")
    if not answer:
        return {"scope_flagged": False}

    if _llm_flags_scope_violation(state["question"], answer):
        logger.warning("scope_guard_node: answer flagged as out-of-scope, discarding answer + citations")
        return {"answer": SCOPE_VIOLATION_MESSAGE, "citations": [], "scope_flagged": True}

    return {"scope_flagged": False}

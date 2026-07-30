"""
Plan 2 Step 9 (roadmap) / Plan 2 Step 4 (local numbering): Output guardrail —
scope enforcement.

Runs after citation, before update_history. Checks whether the final,
citation-annotated answer stayed grounded in its cited sources, or drifted
into the assistant's own opinions/advice/recommendations not actually
supported by anything it cited. LLM-based (semantic judgment, not something
regex can catch), routed through the SIMPLE lane (task="grade") — same
reasoning as injection_guard_node's fallback: structured classification,
not generation.

If flagged, the answer (and any citations built from it) is discarded and
replaced with a fixed scope-refusal message, rather than silently letting
scope-violating content reach the user with a References list attached that
would misleadingly imply the drifted content was itself source-grounded.

Plan 2 Step 5 update: also sets scope_flagged on state (mirroring
is_injection's pattern) so feedback logging can capture whether this
guardrail fired, without needing to parse log text — detection logic itself
is unchanged from Step 4.

Priority 4 update: this node's definition of "in scope" was widened, NOT
bypassed, now that graph/web_search_node.py can answer out-of-scope
questions via a real web search instead of just refusing. Previously this
node flagged anything outside the 5 KB domains as out of scope, which would
have auto-discarded every web-search answer (defeating the point of Priority
4, since ANY web-search answer is by definition outside the 5 domains).
The prompt now explicitly treats an answer clearly labeled as web-sourced
(web_search_node.py always opens with "Based on a web search: ...") as
in-scope by construction — this guardrail's job is now narrowly "does the
answer stay grounded in whatever it cited (KB or web), or does it drift into
ungrounded opinion/advice," not "is the topic one of the 5 KB domains."
"""
import logging
from langchain_core.prompts import ChatPromptTemplate

from graph.state import RAGState
from llm.task_router import get_llm
from llm.errors import AllProvidersFailedError

logger = logging.getLogger("llm_manager")

SCOPE_VIOLATION_MESSAGE = (
    "I can only provide factual information grounded in my cited sources — "
    "not personal opinions or advice beyond that. Could you rephrase your "
    "question to ask about something factual?"
)

SCOPE_CHECK_PROMPT = """You are reviewing an AI assistant's answer for scope compliance. The assistant
answers factual questions either from its knowledge base (AI policy, climate science, economics, public
health, AI research) or, when a question falls outside that knowledge base, from a web search — answers
of the second kind are always clearly labeled at the start (e.g. "Based on a web search: ..."). Either
kind of answer is IN SCOPE as long as it stays grounded in its cited sources.

Flag an answer as OUT OF SCOPE only if it drifts into the assistant's own opinions, subjective
recommendations, or advice not actually supported by anything it cited — NOT merely because its topic
falls outside the 5 knowledge-base domains, and NOT merely because it's labeled as a web search result.
A web-sourced answer that just reports facts from its cited sources is IN SCOPE.

Question: {question}

Answer: {answer}

Does this answer drift into ungrounded opinion/advice, or does it stay grounded in its cited sources
(KB or web)? Respond with ONLY "yes" (drifts into ungrounded opinion/advice) or "no" (stays grounded).

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
    """Checks the final answer for ungrounded opinion/advice drift. If
    flagged, discards the answer and its citations, replacing both with a
    fixed refusal and setting scope_flagged=True — otherwise passes the
    answer through unchanged with scope_flagged=False."""
    answer = state.get("answer", "")
    if not answer:
        return {"scope_flagged": False}

    if _llm_flags_scope_violation(state["question"], answer):
        logger.warning("scope_guard_node: answer flagged as ungrounded opinion/advice, discarding answer + citations")
        return {"answer": SCOPE_VIOLATION_MESSAGE, "citations": [], "scope_flagged": True}

    return {"scope_flagged": False}

"""
Task-based LLM selection. Graph nodes call get_llm(task=...) instead of
building a ChatGroq (or any provider) directly - this file decides which lane
(and therefore which provider fallback chain) a given task should use.

SIMPLE lane (NVIDIA NIM -> local LM Studio; Groq excluded to protect its quota):
    structured, low-reasoning, mostly JSON-shaped outputs.
    Tasks: "rewrite", "route", "grade"

COMPLEX lane (Groq -> NVIDIA NIM -> local LM Studio):
    real reasoning/synthesis quality across multi-domain context.
    Tasks: "generate", "decompose"

Usage in a node:
    from llm.task_router import get_llm
    llm = get_llm(task="generate", temperature=0)
    chain = prompt | llm
"""
from llm.manager import get_llm_manager
from llm.providers import COMPLEX_PROVIDER_BUILDERS, SIMPLE_PROVIDER_BUILDERS

SIMPLE_TASKS = {"rewrite", "route", "grade", "hyde"}
COMPLEX_TASKS = {"generate", "decompose"}


def get_llm(task: str, temperature: float = 0):
    """Returns the correct lane's LLMManager for the given task name."""
    if task in SIMPLE_TASKS:
        return get_llm_manager(SIMPLE_PROVIDER_BUILDERS, temperature=temperature, lane_name="simple")
    elif task in COMPLEX_TASKS:
        return get_llm_manager(COMPLEX_PROVIDER_BUILDERS, temperature=temperature, lane_name="complex")
    else:
        raise ValueError(
            f"Unknown task '{task}'. Expected one of: {SIMPLE_TASKS | COMPLEX_TASKS}. "
            f"If this is a new node, add its task name to SIMPLE_TASKS or COMPLEX_TASKS "
            f"in llm/task_router.py."
        )

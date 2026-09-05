"""
Factory functions for each supported LLM provider. Each function returns a
configured LangChain chat model instance, or None if that provider isn't
configured (missing API key / package not installed) - so the manager can
skip it cleanly instead of crashing.

Two "lanes" are defined at the bottom of this file, matching different task
needs:
  - COMPLEX_PROVIDER_BUILDERS: for real reasoning/synthesis work (answer
    generation, decompose-and-synthesize). Groq's best model is reserved for
    this lane only, falling back to NVIDIA NIM, then local LM Studio.
  - SIMPLE_PROVIDER_BUILDERS: for structured, low-reasoning, mostly JSON-shaped
    outputs (query rewriting, routing, document grading). Groq is deliberately
    excluded here to protect its quota for the complex lane - NVIDIA NIM is
    primary, falling back to local LM Studio.

To add a new provider later: write one new build_x() function following this
same pattern, then add it to whichever lane(s) make sense below.
"""
import logging
from llm.config import ProviderSettings

logger = logging.getLogger("llm_manager")


def build_groq(settings: ProviderSettings, temperature: float = 0):
    if not settings.groq_api_key:
        logger.warning("Groq not configured (missing GROQ_API_KEY) - skipping")
        return None
    try:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=temperature,
        )
    except ImportError:
        logger.warning("langchain-groq not installed - skipping Groq provider")
        return None


def build_nvidia(settings: ProviderSettings, temperature: float = 0):
    if not settings.nvidia_api_key:
        logger.warning("NVIDIA NIM not configured (missing NVIDIA_API_KEY) - skipping")
        return None
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url=settings.nvidia_base_url,
            api_key=settings.nvidia_api_key,
            model=settings.nvidia_model,
            temperature=temperature,
        )
    except ImportError:
        logger.warning("langchain-openai not installed - skipping NVIDIA NIM provider")
        return None


def build_local(settings: ProviderSettings, temperature: float = 0):
    if not settings.lm_studio_enabled:
        logger.info("Local LM Studio provider disabled via LM_STUDIO_ENABLED - skipping")
        return None
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url=settings.lm_studio_base_url,
            api_key="lm-studio",  # LM Studio ignores the key but the client requires one
            model=settings.lm_studio_model,
            temperature=temperature,
        )
    except ImportError:
        logger.warning("langchain-openai not installed - skipping local LM Studio provider")
        return None


# --- Lane definitions ---
# Each lane is an ordered list of (provider_name, builder_function). The
# manager tries them in this order, failing over left-to-right.

COMPLEX_PROVIDER_BUILDERS = [
    ("groq", build_groq),
    ("nvidia", build_nvidia),
    ("local_lm_studio", build_local),
]

SIMPLE_PROVIDER_BUILDERS = [
    ("nvidia", build_nvidia),
    ("groq", build_groq),
    ("local_lm_studio", build_local),
]

# Kept for backward compatibility / the standalone health_check.py script,
# which checks every provider regardless of lane.
ALL_PROVIDER_BUILDERS = [
    ("groq", build_groq),
    ("nvidia", build_nvidia),
    ("local_lm_studio", build_local),
]

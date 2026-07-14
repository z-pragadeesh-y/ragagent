"""
LLMManager: tries a given ordered list of providers with retry-then-failover
logic. Used via the task-based get_llm(task) helper in llm/task_router.py -
graph nodes should import get_llm from task_router, not LLMManager directly.

For each provider: rate-limit / server errors switch to the next provider
immediately; timeout / network errors retry the SAME provider a few times
(with backoff) before switching; auth/config errors (401/403) raise
immediately, since retrying or switching providers can't fix a bad API key.
If every provider in the list fails, AllProvidersFailedError is raised, and
calling nodes decide how to degrade (e.g. return a friendly message instead
of crashing the whole graph).
"""
import logging
import time
from typing import Any, List, Optional, Tuple, Callable

from langchain_core.runnables import Runnable, RunnableConfig

from llm.config import get_settings, ProviderSettings
from llm.errors import classify_error, ErrorCategory, ProviderConfigError, AllProvidersFailedError

logger = logging.getLogger("llm_manager")


def _configure_logging():
    """Sets up console (and optionally file) logging for the llm_manager logger, once."""
    if logger.handlers:
        return  # already configured

    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    settings = get_settings()
    if settings.log_to_file:
        import os
        os.makedirs(os.path.dirname(settings.log_file_path) or ".", exist_ok=True)
        file_handler = logging.FileHandler(settings.log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


_configure_logging()


class LLMManager(Runnable):
    """A Runnable that wraps a specific ordered list of LLM providers with
    automatic failover. Because it implements LangChain's Runnable interface,
    it can be used directly in a chain (e.g. `prompt | llm_manager`) exactly
    like any single chat model - callers never need special-case code."""

    def __init__(
        self,
        provider_builders: List[Tuple[str, Callable]],
        temperature: float = 0,
        lane_name: str = "default",
    ):
        settings = get_settings()
        self.lane_name = lane_name
        self.max_retries_per_provider = settings.max_retries_per_provider
        self.retry_backoff_seconds = settings.retry_backoff_seconds

        self.providers = []
        for name, builder in provider_builders:
            instance = builder(settings, temperature=temperature)
            if instance is not None:
                self.providers.append((name, instance))

        if not self.providers:
            raise ProviderConfigError(
                f"No LLM providers are configured for the '{lane_name}' lane. "
                f"Check your .env file for the relevant API keys, or make sure "
                f"local LM Studio's server is running."
            )

        logger.info(
            f"LLMManager['{lane_name}'] initialized with providers: "
            f"{[name for name, _ in self.providers]}"
        )

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs) -> Any:
        last_exception: Optional[Exception] = None

        for provider_name, provider_llm in self.providers:
            for attempt in range(1, self.max_retries_per_provider + 1):
                try:
                    result = provider_llm.invoke(input, config=config, **kwargs)
                    if attempt > 1:
                        logger.info(f"[{self.lane_name}] Succeeded with '{provider_name}' on retry {attempt}")
                    else:
                        logger.info(f"[{self.lane_name}] Succeeded with '{provider_name}'")
                    return result

                except Exception as exc:
                    category = classify_error(exc)
                    last_exception = exc

                    if category == ErrorCategory.AUTH_CONFIG:
                        logger.error(f"[{self.lane_name}] '{provider_name}' failed - auth/config error: {exc}")
                        raise ProviderConfigError(
                            f"Provider '{provider_name}' rejected the request due to an auth/config "
                            f"problem (bad or missing API key). Fix this in .env - it will not resolve "
                            f"itself via retry or failover."
                        ) from exc

                    if category in (ErrorCategory.TIMEOUT, ErrorCategory.NETWORK):
                        logger.warning(
                            f"[{self.lane_name}] '{provider_name}' hit a {category.value} error "
                            f"(attempt {attempt}/{self.max_retries_per_provider}): {exc}"
                        )
                        if attempt < self.max_retries_per_provider:
                            time.sleep(self.retry_backoff_seconds * attempt)
                            continue  # retry the same provider
                        else:
                            logger.warning(f"[{self.lane_name}] '{provider_name}' exhausted retries, moving on")
                            break  # move to next provider

                    # RATE_LIMIT, SERVER_ERROR, UNKNOWN: no point retrying the same
                    # provider, move to the next one immediately
                    logger.warning(
                        f"[{self.lane_name}] '{provider_name}' failed ({category.value}): {exc}. "
                        f"Trying next provider."
                    )
                    break

        logger.error(f"[{self.lane_name}] All configured providers failed to produce a response.")
        raise AllProvidersFailedError(
            f"Every configured provider in the '{self.lane_name}' lane failed to respond."
        ) from last_exception


# Cache manager instances by (lane_name, temperature) so we don't rebuild
# provider chains (and re-read .env) on every single node call.
_manager_cache: dict = {}


def get_llm_manager(
    provider_builders: List[Tuple[str, Callable]],
    temperature: float = 0,
    lane_name: str = "default",
) -> LLMManager:
    """Returns a cached LLMManager for the given lane + temperature, building one if needed."""
    key = (lane_name, temperature)
    if key not in _manager_cache:
        _manager_cache[key] = LLMManager(provider_builders, temperature=temperature, lane_name=lane_name)
    return _manager_cache[key]


def check_provider_health() -> dict:
    """Pings every configured provider with a trivial prompt and reports status.
    Useful as a manual diagnostic - NOT called automatically on every request,
    since that would waste quota. Run this yourself when you want to check
    which providers are currently reachable."""
    from llm.providers import ALL_PROVIDER_BUILDERS

    settings = get_settings()
    results = {}
    for name, builder in ALL_PROVIDER_BUILDERS:
        instance = builder(settings, temperature=0)
        if instance is None:
            results[name] = "not_configured"
            continue
        try:
            instance.invoke("Reply with the single word: ok")
            results[name] = "healthy"
        except Exception as exc:
            results[name] = f"unhealthy ({classify_error(exc).value}): {exc}"
    return results

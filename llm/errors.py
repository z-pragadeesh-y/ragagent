"""
Error classification for LLM provider failures.

Different provider SDKs (groq, google-generativeai, openai) raise different
exception classes for the same underlying problem (e.g. HTTP 429). Rather than
importing and special-casing every SDK's exception types individually, we
classify errors generically by inspecting common attributes (status_code) and
the exception class name. This keeps the manager decoupled from any single
provider's SDK internals, so adding a new provider later never requires
touching this file.
"""
from enum import Enum


class ErrorCategory(str, Enum):
    RATE_LIMIT = "rate_limit"       # 429 - switch to next provider immediately
    TIMEOUT = "timeout"             # retry same provider a few times, then switch
    NETWORK = "network"             # retry same provider a few times, then switch
    AUTH_CONFIG = "auth_config"     # 401/403 - configuration problem, do not retry/failover
    SERVER_ERROR = "server_error"   # 5xx - switch to next provider
    UNKNOWN = "unknown"             # log and switch to next provider, do not retry


class ProviderConfigError(Exception):
    """Raised when a provider fails due to bad credentials/configuration (401/403).
    This is never silently failed-over, since retrying or switching providers
    won't fix a configuration problem - it needs a human to fix .env."""
    pass


class AllProvidersFailedError(Exception):
    """Raised when every configured provider failed to produce a response."""
    pass


def classify_error(exc: Exception) -> ErrorCategory:
    """Inspects an exception from any provider SDK and classifies it generically."""
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)

    class_name = type(exc).__name__.lower()

    if status_code == 429 or "ratelimit" in class_name or "resourceexhausted" in class_name:
        return ErrorCategory.RATE_LIMIT

    if status_code in (401, 403) or "authenticat" in class_name or "permissiondenied" in class_name:
        return ErrorCategory.AUTH_CONFIG

    if status_code is not None and 500 <= status_code < 600:
        return ErrorCategory.SERVER_ERROR
    if "internalservererror" in class_name or "serviceunavailable" in class_name:
        return ErrorCategory.SERVER_ERROR

    if "timeout" in class_name:
        return ErrorCategory.TIMEOUT

    if "connect" in class_name or "network" in class_name:
        return ErrorCategory.NETWORK

    return ErrorCategory.UNKNOWN

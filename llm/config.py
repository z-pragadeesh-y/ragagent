"""
Central configuration for all LLM providers. Everything is read from environment
variables (.env), so adding/removing/reconfiguring a provider never requires
touching code - only .env.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ProviderSettings:
    # --- Groq (complex-lane primary) ---
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # --- NVIDIA NIM (simple-lane primary / complex-lane 2nd fallback) ---
    # OpenAI-compatible endpoint - free tier at https://build.nvidia.com
    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "")
    nvidia_base_url: str = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    nvidia_model: str = os.getenv("NVIDIA_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5")

    # --- Local LLM via LM Studio (last resort for both lanes) ---
    # LM Studio exposes an OpenAI-compatible server. Start it from
    # LM Studio's "Local Server" tab before relying on this provider.
    lm_studio_base_url: str = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    lm_studio_model: str = os.getenv("LM_STUDIO_MODEL", "qwen/qwen3.5-9b:2")
    lm_studio_enabled: bool = os.getenv("LM_STUDIO_ENABLED", "true").lower() == "true"

    # --- Failover behavior ---
    max_retries_per_provider: int = int(os.getenv("LLM_MAX_RETRIES_PER_PROVIDER", "2"))
    retry_backoff_seconds: float = float(os.getenv("LLM_RETRY_BACKOFF_SECONDS", "1.5"))

    # --- Logging ---
    log_to_file: bool = os.getenv("LLM_LOG_TO_FILE", "false").lower() == "true"
    log_file_path: str = os.getenv("LLM_LOG_FILE_PATH", "logs/llm_manager.log")

    # --- Tavily (Priority 4: web search fallback for out-of-scope questions) ---
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")


def get_settings() -> ProviderSettings:
    """Returns a fresh settings snapshot from current environment variables."""
    return ProviderSettings()

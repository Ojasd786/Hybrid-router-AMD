"""
config.py
=========

Central configuration module for the Hybrid AI Router.

Responsibilities:
- Load environment variables securely.
- Validate required configuration precisely when needed (Lazy Loading).
- Expose immutable application settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv

# Load .env only for local development.
# During hackathon evaluation the environment variables
# are injected automatically by the harness.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """
    Immutable application settings.
    All values come from environment variables.
    """
    fireworks_api_key: str
    fireworks_base_url: str
    allowed_models: List[str]

    request_timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.1
    max_tokens: int = 4096


class ConfigurationError(RuntimeError):
    """Raised when mandatory configuration is missing."""
    pass


def _require_env(name: str) -> str:
    """
    Read an environment variable.

    Raises:
        ConfigurationError: If the variable does not exist or is empty.
    """
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        raise ConfigurationError(f"Missing required environment variable: {name}")
    return value.strip()


def load_settings() -> Settings:
    """
    Build the application Settings object.
    """
    api_key = _require_env("FIREWORKS_API_KEY")
    base_url = _require_env("FIREWORKS_BASE_URL")
    allowed_models_raw = _require_env("ALLOWED_MODELS")

    models = [model.strip() for model in allowed_models_raw.split(",") if model.strip()]
    if not models:
        raise ConfigurationError("ALLOWED_MODELS contains no valid model IDs.")

    return Settings(
        fireworks_api_key=api_key,
        fireworks_base_url=base_url,
        allowed_models=models,
    )


class LazySettings:
    """
    A lazy-loading proxy that defers parsing environment variables 
    until the first time a setting is actually accessed.
    This prevents fatal import-time crashes in Docker containers.
    """
    def __init__(self):
        self._instance: Optional[Settings] = None

    @property
    def _settings(self) -> Settings:
        if self._instance is None:
            self._instance = load_settings()
        return self._instance

    def __getattr__(self, name: str):
        # Pass through any attribute requests to the underlying Settings object
        return getattr(self._settings, name)


# Export the lazy singleton instead of evaluating immediately
settings = LazySettings()

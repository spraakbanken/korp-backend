"""Utility functions used in tests for Korp configuration."""

from typing import Any

from korp.config import Settings


def get_korp_config() -> dict[str, Any]:
    """Return the resolved Korp settings as a dictionary.

    Settings are loaded the same way as in the application (including
    overrides from `.env` via `Settings`).
    """
    return Settings().model_dump()


def get_test_settings(**overrides: Any) -> Settings:
    """Return Settings for tests without loading .env."""
    return Settings(**{"_env_file": None, **overrides})

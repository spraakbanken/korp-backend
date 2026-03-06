"""Plugin base class for the Korp API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from korp.config import settings


class Plugin(APIRouter):
    """Simple plugin class compatible with FastAPI's router API."""

    def __init__(self, name: str, import_name: str, **kwargs: Any) -> None:
        """Initialize plugin.

        Args:
            name: Plugin name.
            import_name: Plugin import name.
            **kwargs: Additional keyword arguments for APIRouter.
        """
        super().__init__(**kwargs)
        self.name = name
        self.import_name = import_name

    def config(self, key: str, default: Any = None) -> Any:
        """Get plugin configuration value.

        Args:
            key: Configuration key.
            default: Default value if key is not found.

        Returns:
            The configuration value or default.
        """
        return settings.PLUGINS_CONFIG.get(self.import_name, {}).get(key, default)

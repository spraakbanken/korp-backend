"""Database helper module."""
from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from korp.config import Settings

_ESCAPE_RE = re.compile(r"[\x00\n\r\x1a\\'\"]")
_ESCAPE_MAP = {
    "\0": "\\0",
    "\n": "\\n",
    "\r": "\\r",
    "\x1a": "\\Z",
    "'": "\\'",
    '"': '\\"',
    "\\": "\\\\",
}


class MySQL:
    """Database helper with async SQLAlchemy engine."""

    def __init__(self) -> None:
        """Initialize MySQL helper."""
        self._async_engine: AsyncEngine | None = None

    def init_app(self, settings: Settings) -> None:
        """Initialize async engine from settings."""
        async_url = URL.create(
            "mysql+asyncmy",
            username=settings.DB_USER or None,
            password=settings.DB_PASSWORD or None,
            host=settings.DB_HOST,
            port=int(settings.DB_PORT),
            database=settings.DB_NAME or None,
            query={"charset": settings.DB_CHARSET},
        )
        self._async_engine = create_async_engine(async_url, pool_pre_ping=True, pool_recycle=3600)

    def _require_async_engine(self) -> AsyncEngine:
        if self._async_engine is None:
            raise RuntimeError("MySQL not initialized. Call mysql.init_app(settings) first.")
        return self._async_engine

    @asynccontextmanager
    async def async_connection(self) -> AsyncIterator[AsyncConnection]:
        """Yield an async database connection."""
        engine = self._require_async_engine()
        async with engine.connect() as conn:
            yield conn

    async def dispose_async(self) -> None:
        """Dispose of the async engine and its connection pool."""
        if self._async_engine is not None:
            await self._async_engine.dispose()

    @staticmethod
    def escape_string(value: str) -> str:
        """Escape a string for safe use in SQL queries.

        Args:
            value: The string to escape.

        Returns:
            The escaped string.
        """
        return _ESCAPE_RE.sub(lambda m: _ESCAPE_MAP[m.group(0)], value)


mysql = MySQL()

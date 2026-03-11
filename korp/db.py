"""Database helper module."""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from logging import getLogger
from time import perf_counter
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from korp.config import Settings

logger = getLogger(__name__)

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

_QUERY_START_TIME_KEY = "_korp_db_query_started_at"
_SQL_TRUNCATION_SUFFIX = "..."


def _compact_sql(statement: str, max_length: int) -> str:
    """Compact SQL into a single line and optionally truncate for logging.

    Args:
        statement: The SQL statement to compact.
        max_length: Maximum length of the returned string. 0 or negative means no limit.

    Returns:
        The compacted SQL string, optionally truncated.
    """
    compact = " ".join(statement.split())
    if max_length > 0 and len(compact) > max_length:
        return compact[: max(0, max_length - len(_SQL_TRUNCATION_SUFFIX))] + _SQL_TRUNCATION_SUFFIX
    return compact


class MySQL:
    """Database helper with async SQLAlchemy engine."""

    __slots__ = ("_async_engine",)

    def __init__(self) -> None:
        """Initialize MySQL helper."""
        self._async_engine: AsyncEngine | None = None

    def init(self, settings: Settings) -> None:
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
        connect_args: dict[str, int] = {}
        if settings.DB_CONNECT_TIMEOUT > 0:
            connect_args["connect_timeout"] = settings.DB_CONNECT_TIMEOUT
        if settings.DB_READ_TIMEOUT > 0:
            connect_args["read_timeout"] = settings.DB_READ_TIMEOUT

        engine_kwargs: dict[str, Any] = {
            "pool_pre_ping": True,
            "pool_recycle": 3600,
            "pool_timeout": max(1, settings.DB_POOL_TIMEOUT),
        }
        if connect_args:
            engine_kwargs["connect_args"] = connect_args

        self._async_engine = create_async_engine(async_url, **engine_kwargs)
        self._register_query_logging(
            slow_query_seconds=max(0.0, settings.DB_SLOW_QUERY_SECONDS),
            sql_max_length=max(0, settings.DB_LOG_SQL_MAX_LENGTH),
        )

    def _register_query_logging(self, *, slow_query_seconds: float, sql_max_length: int) -> None:
        """Register SQLAlchemy event listeners for query timing/error logging."""
        engine = self._require_async_engine()

        if slow_query_seconds > 0:

            @event.listens_for(engine.sync_engine, "before_cursor_execute")
            def before_cursor_execute(
                conn: Any,
                cursor: Any,
                statement: str,
                parameters: Any,
                context: Any,
                executemany: bool,
            ) -> None:
                del cursor, statement, parameters, context, executemany
                conn.info[_QUERY_START_TIME_KEY] = perf_counter()

            @event.listens_for(engine.sync_engine, "after_cursor_execute")
            def after_cursor_execute(
                conn: Any,
                cursor: Any,
                statement: str,
                parameters: Any,
                context: Any,
                executemany: bool,
            ) -> None:
                del cursor, parameters, context, executemany
                started_at = conn.info.pop(_QUERY_START_TIME_KEY, None)
                if started_at is None:
                    return
                elapsed = perf_counter() - started_at
                if elapsed >= slow_query_seconds:
                    logger.warning("Slow DB query %.3fs: %s", elapsed, _compact_sql(statement, sql_max_length))

        @event.listens_for(engine.sync_engine, "handle_error")
        def handle_error(exception_context: Any) -> None:
            conn = exception_context.connection
            statement = exception_context.statement or "<unknown SQL>"
            original_exception = exception_context.original_exception

            elapsed_info = ""
            if conn is not None:
                started_at = conn.info.pop(_QUERY_START_TIME_KEY, None)
                if started_at is not None:
                    elapsed = perf_counter() - started_at
                    elapsed_info = f" after {elapsed:.3f}s"

            logger.warning(
                "DB query failed%s: %s; error=%r",
                elapsed_info,
                _compact_sql(statement, sql_max_length),
                original_exception,
            )

    def _require_async_engine(self) -> AsyncEngine:
        if self._async_engine is None:
            raise RuntimeError("MySQL not initialized. Call init_app(settings) first.")
        return self._async_engine

    @asynccontextmanager
    async def async_connection(self) -> AsyncIterator[AsyncConnection]:
        """Yield an async database connection."""
        engine = self._require_async_engine()
        async with engine.connect() as conn:
            try:
                yield conn
            except BaseException:
                # On error/cancellation, invalidate to avoid reusing a potentially broken connection
                with contextlib.suppress(BaseException):
                    await asyncio.shield(conn.invalidate())
                raise

    async def close(self) -> None:
        """Dispose of the async engine and its connection pool."""
        # Avoid race conditions by setting the engine to None before disposing
        engine = self._async_engine
        self._async_engine = None
        if engine is not None:
            await engine.dispose()


def escape_string(value: str) -> str:
    """Escape a string for safe use in SQL queries.

    Args:
        value: The string to escape.

    Returns:
        The escaped string.
    """
    return _ESCAPE_RE.sub(lambda m: _ESCAPE_MAP[m.group(0)], value)

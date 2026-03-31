"""Rate-limiting helpers for API routes."""

# ruff: noqa: PLC0415
from __future__ import annotations

import asyncio
import inspect
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.routing import APIRoute

if TYPE_CHECKING:
    from limits import RateLimitItem
    from limits.aio.strategies import RateLimiter
    from limits.storage import StorageTypes

    from korp.config import Settings

_HEADER_FIELDS: tuple[tuple[str, str], ...] = (
    ("policy", "X-RateLimit-Policy"),
    ("limit", "X-RateLimit-Limit"),
    ("remaining", "X-RateLimit-Remaining"),
    ("reset_seconds", "X-RateLimit-Reset"),
    ("retry_after_seconds", "Retry-After"),
)


@dataclass(frozen=True)
class RateLimitCheck:
    """Result of a rate-limit check."""

    allowed: bool
    policy: str | None = None
    limit: int | None = None
    remaining: int | None = None
    reset_seconds: int | None = None
    retry_after_seconds: int | None = None

    @property
    def headers(self) -> dict[str, str]:
        """Return HTTP headers representing the check result."""
        return {header: str(value) for attr, header in _HEADER_FIELDS if (value := getattr(self, attr)) is not None}


class RequestRateLimiter:
    """Thin wrapper around `limits` for request-based rate limiting."""

    def __init__(
        self,
        *,
        storage: StorageTypes,
        strategy: RateLimiter,
        parse_limit: Callable[[str], list[RateLimitItem]],
        headers_mode: str,
    ) -> None:
        """Store the provided storage and strategy instances."""
        self._storage = storage
        self._strategy = strategy
        self._parse_limit = parse_limit
        self._headers_mode = headers_mode
        self._parsed_limits: dict[str, list[RateLimitItem]] = {}

    @classmethod
    async def create(cls, storage_uri: str, *, headers_mode: str) -> RequestRateLimiter:
        """Create a limiter instance for the provided storage backend URI.

        Args:
            storage_uri: `limits.aio` storage URI, for example `async+memcached://127.0.0.1:11211`.
            headers_mode: Header mode (`none`, `on_reject`, or `always`).

        Returns:
            An initialized request limiter.

        Raises:
            RuntimeError: If required dependencies for rate limiting are not installed.
        """
        try:
            from limits import parse_many
            from limits.aio.strategies import FixedWindowRateLimiter
            from limits.storage import storage_from_string
        except ImportError:
            raise RuntimeError(
                "Rate limiting dependencies are not installed. Please install the 'rate-limiting' optional "
                "dependencies or disable rate limiting in the settings."
            ) from None
        storage = storage_from_string(storage_uri)
        strategy = FixedWindowRateLimiter(storage)
        return cls(storage=storage, strategy=strategy, parse_limit=parse_many, headers_mode=headers_mode)

    def _get_limit_items(self, limit: str) -> list[RateLimitItem]:
        if limit not in self._parsed_limits:
            self._parsed_limits[limit] = self._parse_limit(limit)
        return self._parsed_limits[limit]

    @staticmethod
    def _pick_header_window(windows: list[tuple[int, int, int]]) -> tuple[int, int, int]:
        """Pick the most restrictive window for response headers.

        Args:
            windows: List of tuples `(limit, remaining, reset_seconds)`.

        Returns:
            The most restrictive tuple.
        """
        return min(
            windows,
            key=lambda w: (
                w[1] / w[0] if w[0] > 0 else 1.0,
                w[1],
                w[2],
            ),
        )

    @staticmethod
    def _client_identifier(request: Request) -> str:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            first_hop = forwarded_for.split(",", 1)[0].strip()
            if first_hop:
                return first_hop
        if request.client and request.client.host:
            return request.client.host
        return "unknown"

    @staticmethod
    def _route_identifier(request: Request) -> str:
        route = request.scope.get("route")
        if isinstance(route, APIRoute):
            return route.path
        return request.scope.get("path", "")

    @staticmethod
    def _extract_window(stats: object, item: RateLimitItem, now: float) -> tuple[int, int, int]:
        """Extract (limit, remaining, reset_seconds) from a stats object.

        Returns:
            A tuple of (limit, remaining, reset_seconds).
        """
        return (
            item.amount,
            max(0, int(getattr(stats, "remaining", 0))),
            max(0, math.ceil(getattr(stats, "reset_time", now))),
        )

    async def check_request(self, request: Request, *, limit: str) -> RateLimitCheck:
        """Consume quota for a request and return whether it is allowed.

        Args:
            request: Incoming HTTP request.
            limit: Rate limit expression, for example `60/minute`.

        Returns:
            The result of the rate-limit check, including retry time when blocked.
        """
        items = self._get_limit_items(limit)
        route = self._route_identifier(request)
        method = request.method
        client = self._client_identifier(request)
        identifiers = ("korp", route, method, client)

        for item in items:
            allowed = await self._strategy.hit(item, *identifiers)
            if not allowed:
                stats = await self._strategy.get_window_stats(item, *identifiers)
                now = time.time()
                _, remaining, reset_seconds = self._extract_window(stats, item, now)
                retry_after = max(0, math.ceil(getattr(stats, "reset_time", now) - now))
                if self._headers_mode == "none":
                    return RateLimitCheck(allowed=False, retry_after_seconds=retry_after)
                return RateLimitCheck(
                    allowed=False,
                    policy=limit,
                    limit=item.amount,
                    remaining=remaining,
                    reset_seconds=reset_seconds,
                    retry_after_seconds=retry_after,
                )

        if self._headers_mode != "always" or not items:
            return RateLimitCheck(allowed=True)

        stats_results = await asyncio.gather(*(self._strategy.get_window_stats(item, *identifiers) for item in items))
        now = time.time()
        windows = [self._extract_window(stats, item, now) for item, stats in zip(items, stats_results, strict=True)]

        header_limit, header_remaining, header_reset = self._pick_header_window(windows)
        return RateLimitCheck(
            allowed=True,
            policy=limit,
            limit=header_limit,
            remaining=header_remaining,
            reset_seconds=header_reset,
        )

    async def close(self) -> None:
        """Close the underlying storage backend, if supported."""
        close_method = getattr(self._storage, "close", None)
        if close_method is None:
            return
        result = close_method()
        if inspect.isawaitable(result):
            await result


def resolve_rate_limit_storage_uri(settings: Settings) -> str | None:
    """Resolve the storage URI to use for rate limiting.

    Args:
        settings: Application settings.

    Returns:
        The configured storage URI, an URI derived from `MEMCACHED_SERVER`, or `None`.
    """
    if settings.RATE_LIMIT_STORAGE_URI:
        if settings.RATE_LIMIT_STORAGE_URI.startswith("memcached://"):
            return f"async+{settings.RATE_LIMIT_STORAGE_URI}"
        return settings.RATE_LIMIT_STORAGE_URI
    if settings.MEMCACHED_SERVER:
        return f"async+memcached://{settings.MEMCACHED_SERVER}"
    return None


def resolve_rate_limit(route_path: str, *, settings: Settings) -> str | None:
    """Resolve the effective rate limit for a route.

    Resolution order (highest priority first):
    1. Per-route override from `settings.RATE_LIMITS` (empty string disables the limit).
    2. Global default from `settings.RATE_LIMIT_DEFAULT`.

    Args:
        route_path: The route path, e.g. `"/query"`.
        settings: Application settings.

    Returns:
        The resolved rate limit string, or `None` if no limit is configured.
    """
    normalized = route_path.lstrip("/")
    keys = (route_path, normalized) if normalized != route_path else (normalized,)

    # Check per-route overrides (support both "/query" and "query" as keys)
    for key in keys:
        if key in settings.RATE_LIMITS:
            return settings.RATE_LIMITS[key] or None  # Empty string disables

    # Fall back to global default
    return settings.RATE_LIMIT_DEFAULT or None

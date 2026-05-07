"""Unit tests for rate-limit helpers."""

from __future__ import annotations

from dataclasses import dataclass

import anyio
import pytest
from pydantic import ValidationError
from starlette.requests import Request

from korp.rate_limit import RateLimitCheck, RequestRateLimiter, resolve_rate_limit, resolve_rate_limit_storage_uri
from tests.configutils import get_test_settings

_RESET_SECOND = 1_000_000_000


def test_resolve_rate_limit_storage_uri_prefers_explicit_value() -> None:
    """Test that explicit RATE_LIMIT_STORAGE_URI takes precedence."""
    settings = get_test_settings(
        RATE_LIMIT_STORAGE_URI="memcached://custom.example:11211",
        MEMCACHED_SERVER="fallback.example:11211",
    )
    assert resolve_rate_limit_storage_uri(settings) == "async+memcached://custom.example:11211"


def test_resolve_rate_limit_storage_uri_preserves_async_uri() -> None:
    """Test that explicit async storage URIs remain unchanged."""
    settings = get_test_settings(
        RATE_LIMIT_STORAGE_URI="async+memcached://custom.example:11211",
        MEMCACHED_SERVER="fallback.example:11211",
    )
    assert resolve_rate_limit_storage_uri(settings) == "async+memcached://custom.example:11211"


def test_resolve_rate_limit_storage_uri_falls_back_to_memcached_server() -> None:
    """Test that MEMCACHED_SERVER is used when no explicit storage URI is set."""
    settings = get_test_settings(
        RATE_LIMIT_STORAGE_URI=None,
        MEMCACHED_SERVER="cache.example:11211",
    )
    assert resolve_rate_limit_storage_uri(settings) == "async+memcached://cache.example:11211"


def test_resolve_rate_limit_storage_uri_returns_none_without_config() -> None:
    """Test that no storage URI is returned when neither explicit URI nor MEMCACHED_SERVER is configured."""
    settings = get_test_settings(
        RATE_LIMIT_STORAGE_URI=None,
        MEMCACHED_SERVER=None,
    )
    assert resolve_rate_limit_storage_uri(settings) is None


def test_rate_limit_check_headers_include_all_available_values() -> None:
    """Test that rate-limit headers include all available values."""
    check = RateLimitCheck(
        allowed=False,
        limit=10,
        remaining=0,
        reset_seconds=1_000_000_000,
        retry_after_seconds=8,
    )
    assert check.headers == {
        "X-RateLimit-Limit": "10",
        "X-RateLimit-Remaining": "0",
        "X-RateLimit-Reset": "1000000000",
        "Retry-After": "8",
    }


def test_rate_limit_headers_setting_rejects_invalid_value() -> None:
    """Test that unsupported RATE_LIMIT_HEADERS values are rejected."""
    with pytest.raises(ValidationError):
        get_test_settings(RATE_LIMIT_HEADERS="sometimes")


@dataclass(frozen=True)
class _FakeLimit:
    name: str
    amount: int
    expiry: int = 60

    def get_expiry(self) -> int:
        return self.expiry


@dataclass(frozen=True)
class _FakeWindowStats:
    reset_time: float
    remaining: int


class _FakeStrategy:
    def __init__(self, hits: dict[str, bool], stats: dict[str, _FakeWindowStats]) -> None:
        self._hits = hits
        self._stats = stats
        self.hit_calls: list[str] = []
        self.stats_calls: list[str] = []

    async def hit(self, item: _FakeLimit, *identifiers: str) -> bool:
        del identifiers
        self.hit_calls.append(item.name)
        return self._hits[item.name]

    async def get_window_stats(self, item: _FakeLimit, *identifiers: str) -> _FakeWindowStats:
        del identifiers
        self.stats_calls.append(item.name)
        return self._stats[item.name]


def _fake_parse_many(limit_string: str) -> list[_FakeLimit]:
    assert limit_string == "1/second;10/minute"
    return [_FakeLimit("per_second", amount=1), _FakeLimit("per_minute", amount=10)]


def _make_request() -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/limited",
            "raw_path": b"/limited",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        }
    )


def test_multi_limit_always_mode_uses_most_restrictive_window_for_headers() -> None:
    """Test that `always` mode reports the most restrictive active window."""

    async def _run() -> RateLimitCheck:
        strategy = _FakeStrategy(
            hits={"per_second": True, "per_minute": True},
            stats={
                "per_second": _FakeWindowStats(reset_time=_RESET_SECOND, remaining=0),
                "per_minute": _FakeWindowStats(reset_time=1_000_000_060, remaining=9),
            },
        )
        limiter = RequestRateLimiter(
            storage=object(),  # type: ignore
            strategy=strategy,  # type: ignore
            parse_limit=_fake_parse_many,  # type: ignore
            headers_mode="always",
        )
        check = await limiter.check_request(_make_request(), limit="1/second;10/minute")
        assert strategy.hit_calls == ["per_second", "per_minute"]
        assert strategy.stats_calls == ["per_second", "per_minute"]
        return check

    check = anyio.run(_run)
    assert check.allowed
    assert check.policy == "1/second;10/minute"
    assert check.limit == 1
    assert check.remaining == 0
    assert check.reset_seconds == _RESET_SECOND


def test_multi_limit_reject_short_circuits_after_first_failed_limit() -> None:
    """Test that limit evaluation stops after the first failed limit."""

    async def _run() -> tuple[RateLimitCheck, _FakeStrategy]:
        strategy = _FakeStrategy(
            hits={"per_second": False, "per_minute": True},
            stats={
                "per_second": _FakeWindowStats(reset_time=1_000_000_010, remaining=0),
                "per_minute": _FakeWindowStats(reset_time=1_000_000_060, remaining=9),
            },
        )
        limiter = RequestRateLimiter(
            storage=object(),  # type: ignore
            strategy=strategy,  # type: ignore
            parse_limit=_fake_parse_many,  # type: ignore
            headers_mode="on_reject",
        )
        return await limiter.check_request(_make_request(), limit="1/second;10/minute"), strategy

    check, strategy = anyio.run(_run)
    assert not check.allowed
    assert check.policy == "1/second;10/minute"
    assert check.limit == 1
    assert check.remaining == 0
    assert check.retry_after_seconds is not None
    assert strategy.hit_calls == ["per_second"]
    assert strategy.stats_calls == ["per_second"]


# --- resolve_rate_limit tests ---


def test_resolve_rate_limit_returns_none_when_no_config() -> None:
    """Test that no effective limit is resolved without rate-limit config."""
    s = get_test_settings()
    assert resolve_rate_limit("/query", settings=s) is None


def test_resolve_rate_limit_global_default() -> None:
    """Test that RATE_LIMIT_DEFAULT is used as the fallback."""
    s = get_test_settings(RATE_LIMIT_DEFAULT="60/minute")
    assert resolve_rate_limit("/query", settings=s) == "60/minute"


def test_resolve_rate_limit_per_route_overrides_global_default() -> None:
    """Test that per-route RATE_LIMITS override the global default."""
    s = get_test_settings(RATE_LIMIT_DEFAULT="60/minute", RATE_LIMITS={"/query": "30/minute"})
    assert resolve_rate_limit("/query", settings=s) == "30/minute"


def test_resolve_rate_limit_per_route_without_slash() -> None:
    """Test that RATE_LIMITS keys without a leading slash match route paths."""
    s = get_test_settings(RATE_LIMITS={"query": "5/minute"})
    assert resolve_rate_limit("/query", settings=s) == "5/minute"


def test_resolve_rate_limit_empty_per_route_disables() -> None:
    """Test that an empty RATE_LIMITS value disables route rate limiting."""
    s = get_test_settings(RATE_LIMIT_DEFAULT="60/minute", RATE_LIMITS={"/query": ""})
    assert resolve_rate_limit("/query", settings=s) is None


def test_resolve_rate_limit_unmatched_route_uses_default() -> None:
    """Test that routes without a per-route override use RATE_LIMIT_DEFAULT."""
    s = get_test_settings(RATE_LIMIT_DEFAULT="60/minute", RATE_LIMITS={"/query": "30/minute"})
    assert resolve_rate_limit("/other", settings=s) == "60/minute"

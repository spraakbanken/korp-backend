"""Tests for optional per-route rate limiting in `api_handler`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from korp.dependencies import CtxDep
from korp.handler import api_handler
from korp.rate_limit import RateLimitCheck
from tests.configutils import get_test_settings

# HTTP response status codes
HTTP_OK = 200
HTTP_TOO_MANY_REQUESTS = 429


@dataclass
class FakeRateLimiter:
    """Fake app-level rate limiter used for testing."""

    result: RateLimitCheck
    calls: int = 0
    last_limit: str | None = None

    async def check_request(self, _request: Any, *, limit: str) -> RateLimitCheck:
        """Record invocation and return the configured result.

        Returns:
            The configured fake rate-limit check result.
        """
        self.calls += 1
        self.last_limit = limit
        return self.result


def _make_test_app(rate_limiter: FakeRateLimiter | None) -> FastAPI:
    app = FastAPI()
    app.state.cache_enabled = False
    app.state.memcached = object()
    app.state.db = object()
    app.state.cwb = object()
    app.state.rate_limiter = rate_limiter

    @app.get("/open", response_model=None)
    @api_handler
    async def open_route(_ctx: CtxDep) -> dict:
        return {"ok": True}

    @app.get("/limited", response_model=None)
    @api_handler(rate_limit=True)
    async def limited_route(_ctx: CtxDep) -> dict:
        return {"ok": True}

    return app


def test_unlimited_route_skips_rate_limiter() -> None:
    """Test that routes without `rate_limit` skip the app limiter."""
    limiter = FakeRateLimiter(result=RateLimitCheck(allowed=False, retry_after_seconds=12))
    app = _make_test_app(limiter)

    with TestClient(app) as client:
        response = client.get("/open")

    assert response.status_code == HTTP_OK
    assert limiter.calls == 0


def test_limited_route_allows_when_quota_available() -> None:
    """Test that limited routes pass through when quota is available."""
    from unittest.mock import patch  # noqa: PLC0415

    from korp import handler  # noqa: PLC0415

    limiter = FakeRateLimiter(result=RateLimitCheck(allowed=True, limit=2, remaining=1, reset_seconds=1_000_000_000))
    app = _make_test_app(limiter)

    override = get_test_settings(RATE_LIMIT_DEFAULT="2/minute")
    with patch.object(handler, "settings", override), TestClient(app) as client:
        response = client.get("/limited")

    assert response.status_code == HTTP_OK
    assert limiter.calls == 1
    assert limiter.last_limit == "2/minute"
    assert response.headers["x-ratelimit-limit"] == "2"
    assert response.headers["x-ratelimit-remaining"] == "1"
    assert response.headers["x-ratelimit-reset"] == "1000000000"


def test_limited_route_returns_429_when_quota_exceeded() -> None:
    """Test that limited routes return HTTP 429 when no quota remains."""
    from unittest.mock import patch  # noqa: PLC0415

    from korp import handler  # noqa: PLC0415

    limiter = FakeRateLimiter(
        result=RateLimitCheck(
            allowed=False,
            limit=2,
            remaining=0,
            reset_seconds=1_000_000_000,
            retry_after_seconds=5,
        )
    )
    app = _make_test_app(limiter)

    override = get_test_settings(RATE_LIMIT_DEFAULT="2/minute")
    with patch.object(handler, "settings", override), TestClient(app) as client:
        response = client.get("/limited")

    assert response.status_code == HTTP_TOO_MANY_REQUESTS
    assert response.json() == {"detail": "Rate limit exceeded."}
    assert response.headers["retry-after"] == "5"
    assert response.headers["x-ratelimit-limit"] == "2"
    assert response.headers["x-ratelimit-remaining"] == "0"
    assert response.headers["x-ratelimit-reset"] == "1000000000"


def test_limited_route_is_noop_without_config() -> None:
    """Test that limited routes skip the rate limiter without a configured limit."""
    from unittest.mock import patch  # noqa: PLC0415

    from korp import handler  # noqa: PLC0415

    limiter = FakeRateLimiter(result=RateLimitCheck(allowed=False, retry_after_seconds=1))
    app = _make_test_app(limiter)

    override = get_test_settings(RATE_LIMIT_DEFAULT="", RATE_LIMITS={})
    with patch.object(handler, "settings", override), TestClient(app) as client:
        response = client.get("/limited")

    assert response.status_code == HTTP_OK
    assert limiter.calls == 0


def test_limited_route_is_noop_without_app_limiter() -> None:
    """Test that limited routes behave normally without an initialized app-level limiter."""
    app = _make_test_app(rate_limiter=None)

    with TestClient(app) as client:
        response = client.get("/limited")

    assert response.status_code == HTTP_OK


def test_config_override_changes_effective_limit() -> None:
    """Test that RATE_LIMITS overrides change the limit sent to the limiter."""
    from unittest.mock import patch  # noqa: PLC0415

    from korp import handler  # noqa: PLC0415

    limiter = FakeRateLimiter(result=RateLimitCheck(allowed=True))
    app = _make_test_app(limiter)

    override = get_test_settings(RATE_LIMITS={"/limited": "99/hour"})
    with patch.object(handler, "settings", override), TestClient(app) as client:
        client.get("/limited")

    assert limiter.last_limit == "99/hour"


def test_config_override_disables_rate_limit_with_empty_string() -> None:
    """Test that an empty RATE_LIMITS override disables route rate limiting, even if RATE_LIMIT_DEFAULT is set."""
    from unittest.mock import patch  # noqa: PLC0415

    from korp import handler  # noqa: PLC0415

    limiter = FakeRateLimiter(result=RateLimitCheck(allowed=False, retry_after_seconds=1))
    app = _make_test_app(limiter)

    override = get_test_settings(RATE_LIMIT_DEFAULT="10/minute", RATE_LIMITS={"/limited": ""})
    with patch.object(handler, "settings", override), TestClient(app) as client:
        response = client.get("/limited")

    assert response.status_code == HTTP_OK
    assert limiter.calls == 0

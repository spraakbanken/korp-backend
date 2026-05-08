"""Tests for `api_handler` error formatting behavior."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.testclient import TestClient

from korp.dependencies import CtxDep
from korp.handler import api_handler

HTTP_OK = 200


def _make_test_app() -> FastAPI:
    app = FastAPI()
    app.state.cache_enabled = False
    app.state.memcached = object()
    app.state.db = object()
    app.state.cwb = object()
    app.state.rate_limiter = None

    @app.get("/grouped", response_model=None)
    @api_handler
    async def grouped(_ctx: CtxDep) -> AsyncIterator[dict]:
        raise ExceptionGroup(
            "unhandled errors in a TaskGroup",
            [RuntimeError(), ValueError("crash")],
        )
        yield {}

    return app


def test_format_error_unwraps_exception_group() -> None:
    """Grouped task failures should expose the underlying leaf exception in API responses."""
    app = _make_test_app()

    with TestClient(app) as client:
        response = client.get("/grouped")

    assert response.status_code == HTTP_OK
    assert response.json()["ERROR"] == {"type": "ValueError", "value": "crash"}

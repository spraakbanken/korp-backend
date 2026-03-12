"""Compatibility tests for `api_handler` with newer FastAPI versions."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.routing import APIRoute

from korp.dependencies import CtxDep
from korp.handler import api_handler


def _make_test_app() -> FastAPI:
    app = FastAPI()

    @app.get("/streamed", response_model=None)
    @api_handler
    async def streamed(_ctx: CtxDep) -> AsyncIterator[dict]:
        yield {"ok": True}

    return app


def test_async_generator_route_not_marked_as_fastapi_json_stream() -> None:
    """Decorated async-generator routes should not be marked as JSON streams, since our wrapper is a coroutine."""
    app = _make_test_app()
    route = next(route for route in app.routes if isinstance(route, APIRoute) and route.path == "/streamed")

    assert route.dependant.is_async_gen_callable is False
    assert route.is_json_stream is False

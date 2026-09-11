"""Tests for the shared API error response behavior."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from korp import cqp, handler
from korp.api.schemas import ErrorResponse
from korp.dependencies import CtxDep
from korp.handler import api_handler, install_error_handlers

HTTP_BAD_REQUEST = 400
HTTP_INTERNAL_SERVER_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503
HTTP_UNPROCESSABLE_ENTITY = 422


def _make_test_app() -> FastAPI:
    app = FastAPI()
    app.state.cache_enabled = False
    app.state.memcached = object()
    app.state.db = object()
    app.state.cwb = object()
    app.state.rate_limiter = None
    install_error_handlers(app)

    @app.get("/grouped-cqp-error", response_model=None)
    @api_handler
    async def grouped_cqp_error(_ctx: CtxDep) -> dict:
        raise ExceptionGroup(
            "unhandled errors in a TaskGroup",
            [cqp.CQPError("Invalid CQP query."), RuntimeError("secondary failure")],
        )

    @app.get("/invalid-request", response_model=None)
    @api_handler
    async def invalid_request(_ctx: CtxDep) -> dict:
        raise handler.APIValidationError("Invalid parameter combination.")

    @app.get("/typed", response_model=None)
    @api_handler
    async def typed(_ctx: CtxDep, value: int) -> dict:
        return {"value": value}

    @app.get("/database-error", response_model=None)
    @api_handler
    async def database_error(_ctx: CtxDep) -> dict:
        raise SQLAlchemyError("secret connection details")

    @app.get("/unexpected-error", response_model=None)
    async def unexpected_error() -> None:
        raise RuntimeError("secret implementation details")

    return app


def test_exception_group_uses_the_meaningful_cqp_error() -> None:
    """Ensure that task-group wrappers do not hide a recognized underlying error."""
    app = _make_test_app()

    with TestClient(app) as client:
        response = client.get("/grouped-cqp-error")

    assert response.status_code == HTTP_BAD_REQUEST
    assert response.json() == {
        "code": "cqp_error",
        "title": "CQP query failed",
        "status": HTTP_BAD_REQUEST,
        "detail": "Invalid CQP query.",
    }
    ErrorResponse.model_validate_json(response.content, strict=True)


def test_api_validation_error_uses_the_shared_error_contract() -> None:
    """Ensure that explicit request validation returns the shared HTTP 422 error shape."""
    app = _make_test_app()

    with TestClient(app) as client:
        response = client.get("/invalid-request")

    assert response.status_code == HTTP_UNPROCESSABLE_ENTITY
    assert response.json() == {
        "code": "invalid_request",
        "title": "Invalid request",
        "status": HTTP_UNPROCESSABLE_ENTITY,
        "detail": "Invalid parameter combination.",
    }
    ErrorResponse.model_validate_json(response.content, strict=True)


def test_request_validation_includes_structured_parameter_errors() -> None:
    """Ensure that FastAPI parameter validation is normalized without discarding field diagnostics."""
    app = _make_test_app()

    with TestClient(app) as client:
        response = client.get("/typed", params={"value": "not-an-integer"})

    error = ErrorResponse.model_validate_json(response.content, strict=True)
    assert response.status_code == HTTP_UNPROCESSABLE_ENTITY
    assert error.code == "invalid_request"
    assert error.detail == "Request validation failed."
    assert error.errors is not None
    assert error.errors[0]["loc"] == ["query", "value"]


def test_database_error_hides_backend_details() -> None:
    """Ensure that database failures return a stable service error without leaking connection details."""
    with TestClient(_make_test_app()) as client:
        response = client.get("/database-error")

    assert response.status_code == HTTP_SERVICE_UNAVAILABLE
    assert response.json() == {
        "code": "database_unavailable",
        "title": "Database unavailable",
        "status": HTTP_SERVICE_UNAVAILABLE,
        "detail": "The database backend is unavailable.",
    }


def test_unexpected_error_hides_implementation_details() -> None:
    """Ensure that the app-level fallback returns the public backend error instead of exception text."""
    with TestClient(_make_test_app(), raise_server_exceptions=False) as client:
        response = client.get("/unexpected-error")

    assert response.status_code == HTTP_INTERNAL_SERVER_ERROR
    assert response.json() == {
        "code": "backend_error",
        "title": "Backend error",
        "status": HTTP_INTERNAL_SERVER_ERROR,
        "detail": "An unexpected backend error occurred.",
    }


def test_traceback_requires_server_setting_and_request_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure that a request does not expose a traceback unless the server explicitly allows it."""
    app = _make_test_app()

    with TestClient(app, raise_server_exceptions=False) as client:
        assert "traceback" not in client.get("/unexpected-error?debug=true").json()
        monkeypatch.setattr(handler.settings, "ERROR_TRACEBACKS_ENABLED", True)
        assert "traceback" not in client.get("/unexpected-error").json()
        traceback = client.get("/unexpected-error?debug=true").json()["traceback"]

    assert traceback[-1] == "RuntimeError: secret implementation details"

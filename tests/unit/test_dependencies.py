"""Tests for shared request controls."""

import pytest
from fastapi import FastAPI
from starlette.requests import Request

from korp.api.requests import CommonQueryControls
from korp.config import settings
from korp.dependencies import build_common_params


@pytest.mark.parametrize(
    ("server_cache", "allow_bypass", "request_cache", "expected_cache"),
    [
        (False, False, False, False),
        (False, False, True, False),
        (False, True, False, False),
        (False, True, True, False),
        (True, False, False, True),
        (True, False, True, True),
        (True, True, False, False),
        (True, True, True, True),
    ],
)
def test_cache_policy(
    monkeypatch: pytest.MonkeyPatch,
    server_cache: bool,
    allow_bypass: bool,
    request_cache: bool,
    expected_cache: bool,
) -> None:
    """Only permit request cache bypass when enabled, and never enable unavailable caching."""
    monkeypatch.setattr(settings, "ALLOW_CACHE_BYPASS", allow_bypass)
    app = FastAPI()
    app.state.cache_enabled = server_cache
    request = Request({"type": "http", "app": app})

    common = build_common_params(request, CommonQueryControls.model_validate({"cache": request_cache}))

    assert common.cache is expected_cache

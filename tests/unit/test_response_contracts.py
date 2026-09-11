"""Tests linking public routes to their documentation-only response contracts."""

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute

from korp import app as app_module
from korp.api import schemas
from korp.api.routers import (
    attribute_values,
    cache,
    concordance,
    corpus_config,
    dependency_relations,
    frequencies,
    info,
    lexeme_counts,
    log_likelihood,
    misc,
    routers,
    token_distribution,
)
from korp.dependencies import CtxDep
from korp.handler import api_handler
from tests.configutils import get_test_settings

EXPECTED_SUCCESS_MODELS = {
    "/attribute-values": attribute_values.AttrValuesResponse,
    "/cache": cache.CacheResponse,
    "/concordance": concordance.ConcordanceResponse,
    "/concordance/sample": concordance.ConcordanceSampleResponse,
    "/corpora/config": corpus_config.CorpusConfigResponse,
    "/corpora/info": info.CorpusInfoResponse,
    "/dependency-relations": dependency_relations.RelationsResponse,
    "/dependency-relations/sentences": dependency_relations.RelationsSentencesResponse,
    "/dependency-relations/time": dependency_relations.RelationsResponse,
    "/dependency-relations/time/sentences": dependency_relations.RelationsSentencesResponse,
    "/frequencies": frequencies.FrequenciesResponse,
    "/frequencies/corpus": frequencies.CorpusFrequenciesResponse,
    "/frequencies/time": frequencies.FrequenciesTimeResponse,
    "/health": misc.HealthResponse,
    "/info": info.InfoResponse,
    "/lexeme-counts": lexeme_counts.LexemeCountResponse,
    "/log-likelihood": log_likelihood.LogLikelihoodResponse,
    "/optimize": misc.OptimizeResponse,
    "/token-distribution": token_distribution.TokenDistributionResponse,
}


def _public_get_routes() -> dict[str, APIRoute]:
    """Return schema-visible GET routes keyed by path."""
    routes_by_path = {
        route.path: route
        for api_router in routers
        for route in api_router.routes
        if isinstance(route, APIRoute) and route.include_in_schema and "GET" in route.methods
    }
    assert len(routes_by_path) == len(EXPECTED_SUCCESS_MODELS)
    return routes_by_path


@pytest.mark.parametrize(("path", "success_model"), EXPECTED_SUCCESS_MODELS.items())
def test_public_route_uses_expected_documentation_model(path: str, success_model: type[schemas.ResponseModel]) -> None:
    """Ensure that each public GET route has the expected documented success and error responses."""
    route = _public_get_routes()[path]
    documented_model = route.responses[200]["model"]
    documented_error_model = route.responses[422]["model"]

    assert route.response_model is None
    assert documented_error_model is schemas.ErrorResponse

    assert documented_model is success_model
    has_ndjson = "application/x-ndjson" in route.responses[200].get("content", {})
    assert has_ndjson == (path != "/health")


def test_generated_openapi_contains_every_documented_response_media_type() -> None:
    """Ensure that FastAPI preserves the JSON and NDJSON contracts in the generated OpenAPI document."""
    app = FastAPI()
    for api_router in routers:
        app.include_router(api_router)
    openapi = app.openapi()

    assert set(openapi["paths"]) == set(EXPECTED_SUCCESS_MODELS)
    for path in EXPECTED_SUCCESS_MODELS:
        content = openapi["paths"][path]["get"]["responses"]["200"]["content"]
        assert "application/json" in content
        assert ("application/x-ndjson" in content) == (path != "/health")
        assert "application/json" in openapi["paths"][path]["get"]["responses"]["422"]["content"]


AUTHORIZED_PATHS = {
    "/attribute-values",
    "/concordance",
    "/concordance/sample",
    "/dependency-relations",
    "/dependency-relations/time",
    "/dependency-relations/sentences",
    "/dependency-relations/time/sentences",
    "/frequencies",
    "/frequencies/corpus",
    "/frequencies/time",
    "/lexeme-counts",
    "/log-likelihood",
}


def test_generated_openapi_documents_errors_only_on_applicable_routes() -> None:
    """Ensure that every declared HTTP failure uses the shared JSON error model."""
    app = FastAPI()
    for api_router in routers:
        app.include_router(api_router)

    for path, item in app.openapi()["paths"].items():
        responses = item["get"]["responses"]
        assert ("403" in responses) == (path in AUTHORIZED_PATHS)
        assert "429" not in responses  # A bare FastAPI app should not have rate limiting enabled
        for status in ("400", "422", "500", "503"):
            assert responses[status]["content"] == {
                "application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}
            }
        if path in AUTHORIZED_PATHS:
            assert responses["403"]["content"] == {
                "application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}
            }


@pytest.mark.parametrize(
    ("enabled", "default", "overrides", "expected"),
    [
        (False, "10/minute", {}, False),
        (True, "", {}, False),
        (True, "10/minute", {}, True),
        (True, "", {"/concordance": "2/minute"}, True),
        (True, "", {"concordance": "2/minute"}, True),
        (True, "10/minute", {"/concordance": ""}, False),
    ],
)
def test_openapi_rate_limits_follow_configuration(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, default: str, overrides: dict[str, str], expected: bool
) -> None:
    """Ensure that the OpenAPI document reflects the configured rate limits and exemptions."""
    monkeypatch.setattr(
        app_module,
        "settings",
        get_test_settings(
            CQP_EXECUTABLE="/bin/cqp",
            CWB_SCAN_EXECUTABLE="/bin/cwb-scan-corpus",
            CWB_REGISTRY="/tmp",
            PLUGINS=[],
            RATE_LIMIT_ENABLED=enabled,
            RATE_LIMIT_DEFAULT=default,
            RATE_LIMITS=overrides,
        ),
    )
    app = app_module.create_app()

    @app.get("/exempt")
    @api_handler(rate_limit=False)
    def exempt(_ctx: CtxDep) -> dict:
        return {}

    schema = app.openapi()
    responses = schema["paths"]["/concordance"]["get"]["responses"]
    assert ("429" in responses) == expected
    assert "429" not in schema["paths"]["/exempt"]["get"]["responses"]
    assert "429" not in schema["paths"]["/health"]["get"]["responses"]
    if expected:
        assert set(responses["429"]["content"]) == {"application/json"}
        assert "detail" in responses["429"]["content"]["application/json"]["schema"]["required"]
        assert "Retry-After" in responses["429"]["headers"]
    assert app.openapi() is schema

"""Tests linking public routes to their documentation-only response contracts."""

from types import UnionType
from typing import get_args

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute

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
    """Test that each public GET route has the expected documented success and error responses."""
    route = _public_get_routes()[path]
    documented_model = route.responses[200]["model"]
    documented_error_model = route.responses[422]["model"]

    assert route.response_model is None
    expected_extra = "allow" if success_model is corpus_config.CorpusConfigResponse else "forbid"
    assert success_model.model_config.get("extra") == expected_extra
    assert set(get_args(documented_error_model)) == {
        schemas.PreflightErrorResponse,
        schemas.RequestValidationErrorResponse,
    }

    if path in {"/cache", "/health"}:
        assert documented_model is success_model
        has_ndjson = "application/x-ndjson" in route.responses[200].get("content", {})
        assert has_ndjson == (path == "/cache")
    else:
        assert isinstance(documented_model, UnionType)
        assert set(get_args(documented_model)) == {success_model, schemas.LateJsonErrorResponse}
        assert "application/x-ndjson" in route.responses[200]["content"]


def test_generated_openapi_contains_every_documented_response_media_type() -> None:
    """Test that FastAPI preserves the JSON and NDJSON contracts in the generated OpenAPI document."""
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

"""Tests linking public routes to their documentation-only response contracts."""

import inspect

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute

from korp import app as app_module
from korp.api import schemas
from korp.api.requests import QueryRequestModel
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
from korp.dependencies import CtxDep, QueryCtxDep
from korp.handler import api_handler
from tests.configutils import get_test_settings

EXPECTED_SUCCESS_MODELS = {
    "/attribute-values": attribute_values.AttrValuesResponse,
    "/admin/cache/refresh": cache.CacheResponse,
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


DUAL_METHOD_PATHS = set(EXPECTED_SUCCESS_MODELS) - {"/admin/cache/refresh", "/health", "/info"}
EXPECTED_METHODS = {
    path: ({"GET", "POST"} if path in DUAL_METHOD_PATHS else {"POST"} if path == "/admin/cache/refresh" else {"GET"})
    for path in EXPECTED_SUCCESS_MODELS
}


def _public_routes() -> dict[tuple[str, str], APIRoute]:
    """Return schema-visible routes keyed by path and method."""
    routes = {
        (route.path, method): route
        for api_router in routers
        for route in api_router.routes
        if isinstance(route, APIRoute) and route.methods is not None and route.include_in_schema
        for method in route.methods
    }
    assert len(routes) == sum(map(len, EXPECTED_METHODS.values()))
    return routes


@pytest.mark.parametrize(("path", "success_model"), EXPECTED_SUCCESS_MODELS.items())
def test_public_routes_use_expected_documentation_model(path: str, success_model: type[schemas.ResponseModel]) -> None:
    """Ensure that each public method has the expected documented success and error responses."""
    routes = _public_routes()
    for method in EXPECTED_METHODS[path]:
        route = routes[path, method]
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
    operation_ids = []
    for path, methods in EXPECTED_METHODS.items():
        assert set(openapi["paths"][path]) == {method.lower() for method in methods}
        for method in methods:
            operation = openapi["paths"][path][method.lower()]
            operation_ids.append(operation["operationId"])
            content = operation["responses"]["200"]["content"]
            assert "application/json" in content
            assert ("application/x-ndjson" in content) == (path != "/health")
            assert "application/json" in operation["responses"]["422"]["content"]
            expects_request_body = method == "POST" and path in DUAL_METHOD_PATHS
            assert ("requestBody" in operation) == expects_request_body
            if "requestBody" in operation:
                body_schema = operation["requestBody"]["content"]["application/json"]["schema"]
                component_name = body_schema["$ref"].rsplit("/", maxsplit=1)[-1]
                assert openapi["components"]["schemas"][component_name]["additionalProperties"] is False
    assert len(operation_ids) == len(set(operation_ids))


def test_paired_get_routes_use_one_derived_query_model() -> None:
    """Ensure that GET routes have exactly one query model derived from the `QueryRequestModel`."""
    routes = _public_routes()

    for path in DUAL_METHOD_PATHS:
        route = routes[path, "GET"]
        signature = inspect.signature(route.endpoint)
        ctx_parameter = signature.parameters.get("ctx") or signature.parameters["_ctx"]
        assert ctx_parameter.annotation is QueryCtxDep
        assert len(route.dependant.query_params) == 1
        query_model = route.dependant.query_params[0].field_info.annotation
        assert isinstance(query_model, type)
        assert issubclass(query_model, QueryRequestModel)


def test_paired_get_and_post_fields_have_matching_requirements_and_defaults() -> None:
    """Ensure that derived GET schemas retain the semantic POST model's required fields and defaults."""
    openapi = app_module.create_app({"TESTING": True}).openapi()
    controls = {"cache", "debug", "indent", "stream"}

    for path in DUAL_METHOD_PATHS:
        get_operation = openapi["paths"][path]["get"]
        post_operation = openapi["paths"][path]["post"]
        get_parameters = {
            parameter["name"]: parameter
            for parameter in get_operation["parameters"]
            if parameter["name"] not in controls
        }
        body_ref = post_operation["requestBody"]["content"]["application/json"]["schema"]["$ref"]
        body_schema = openapi["components"]["schemas"][body_ref.rsplit("/", maxsplit=1)[-1]]
        required_body_fields = set(body_schema.get("required", []))

        assert set(get_parameters) == set(body_schema["properties"])
        for name, parameter in get_parameters.items():
            assert parameter["required"] == (name in required_body_fields)
            assert parameter["schema"].get("default") == body_schema["properties"][name].get("default")


def test_representative_request_schema_fields() -> None:
    """Ensure that a representative set of request schemas behave as expected in the generated OpenAPI document."""
    openapi = app_module.create_app({"TESTING": True}).openapi()
    schemas = openapi["components"]["schemas"]
    concordance_request = schemas["ConcordanceRequest"]
    relation_request = schemas["DependencyRelationsRequest"]

    assert concordance_request["required"] == ["corpora", "cqp"]
    assert concordance_request["properties"]["cqp"]["type"] == "array"
    assert concordance_request["properties"]["attributes"]["default"] == ["word"]
    assert relation_request["properties"]["term_type"]["default"] == "word"
    assert set(schemas["LogLikelihoodResponse"]["required"]) >= {"average", "results"}


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
    "/token-distribution",
}


def test_generated_openapi_documents_errors_only_on_applicable_routes() -> None:
    """Ensure that every declared HTTP failure uses the shared JSON error model."""
    app = FastAPI()
    for api_router in routers:
        app.include_router(api_router)

    for path, item in app.openapi()["paths"].items():
        for operation in item.values():
            responses = operation["responses"]
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
    get_responses = schema["paths"]["/concordance"]["get"]["responses"]
    post_responses = schema["paths"]["/concordance"]["post"]["responses"]
    assert ("429" in get_responses) == expected
    assert ("429" in post_responses) == expected
    assert "429" not in schema["paths"]["/exempt"]["get"]["responses"]
    assert "429" not in schema["paths"]["/health"]["get"]["responses"]
    if expected:
        assert set(get_responses["429"]["content"]) == {"application/json"}
        assert "detail" in get_responses["429"]["content"]["application/json"]["schema"]["required"]
        assert "Retry-After" in get_responses["429"]["headers"]
    assert app.openapi() is schema


def test_authorizer_contributes_openapi_security(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured auth plugins should document credentials and operation requirements."""
    monkeypatch.setattr(
        app_module,
        "settings",
        get_test_settings(
            CQP_EXECUTABLE="/bin/cqp",
            CWB_SCAN_EXECUTABLE="/bin/cwb-scan-corpus",
            CWB_REGISTRY="/tmp",
            PLUGINS=["plugins.auth"],
        ),
    )
    schema = app_module.create_app().openapi()

    assert schema["components"]["securitySchemes"]["basicAuth"] == {"type": "http", "scheme": "basic"}
    assert schema["paths"]["/concordance"]["get"]["security"] == [{}, {"basicAuth": []}]
    assert "security" not in schema["paths"]["/health"]["get"]

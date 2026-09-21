"""Utility functions that can be called from tests.

The functions may contain assertions that are subject to rewriting.
"""

from typing import Any, cast

from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from httpx import Response
from pydantic import BaseModel, TypeAdapter, ValidationError

from korp.api.schemas import StreamCompleteEvent, StreamErrorEvent, StreamEvent, StreamResultEvent
from korp.handler import iter_api_route_contexts
from korp.utils import QUERY_DELIM


def get_documented_response_model(client: TestClient, path: str, *, method: str = "GET", status_code: int = 200) -> Any:
    """Return the model attached to an operation's documented response.

    Args:
        client: Test client whose application owns the route.
        path: Route path without a query string.
        method: HTTP method used for the request.
        status_code: Documented response status to inspect.

    Returns:
        The Pydantic-compatible type stored by ``docs_response``.
    """
    app = cast(FastAPI, client.app)
    matching_routes = [
        cast(APIRoute, route_context.original_route)
        for route_context in iter_api_route_contexts(app)
        if route_context.path_format == path and method.upper() in (route_context.methods or ())
    ]
    assert len(matching_routes) == 1, f"Expected one {method} route for {path}, found {len(matching_routes)}"

    response_docs = matching_routes[0].responses.get(status_code, {})
    model = response_docs.get("model")
    assert model is not None, f"{method} {path} has no documented model for HTTP {status_code}"
    return model


def validate_response_contract(client: TestClient, response: Response, path: str, *, method: str = "GET") -> BaseModel:
    """Validate an actual JSON response against the model used by its OpenAPI operation.

    Args:
        client: Test client whose application handled the request.
        response: Serialized HTTP response to validate.
        path: Route path without a query string.
        method: HTTP method used for the request.

    Returns:
        The validated response model instance.

    Raises:
        AssertionError: If the serialized response violates the documented contract.
    """
    model = get_documented_response_model(client, path, method=method, status_code=response.status_code)
    try:
        return TypeAdapter(model).validate_json(response.content, strict=True)
    except ValidationError as exc:
        raise AssertionError(f"{method} {path} response violates its documented contract:\n{exc}") from exc


def validate_ndjson_contract(body: bytes | str, model: type[BaseModel]) -> BaseModel:
    """Validate NDJSON events and their assembled successful result.

    Args:
        body: Complete NDJSON response body.
        model: Ordinary success model for the route-specific result data.

    Returns:
        The assembled and validated route response.

    Raises:
        AssertionError: If the stream reports an error, lacks a successful final event, or violates either contract.
    """
    lines = body.splitlines()
    event_adapter = TypeAdapter(StreamEvent)
    events = [event_adapter.validate_json(line, strict=True) for line in lines if line.strip()]
    assert events, "NDJSON response contains no events"
    assert isinstance(events[-1], StreamCompleteEvent), "NDJSON response does not end with a complete event"
    assert events[-1].ok, "NDJSON response did not complete successfully"
    assert not any(isinstance(event, StreamErrorEvent) for event in events), "NDJSON response contains an error event"

    result: dict[str, Any] = {}
    for event in events:
        if isinstance(event, StreamResultEvent):
            result.update(event.data)
    result["elapsed"] = events[-1].elapsed

    try:
        return model.model_validate(result, strict=True)
    except ValidationError as exc:
        raise AssertionError(f"Assembled NDJSON result violates {model.__name__}:\n{exc}") from exc


def get_response_json(client: TestClient, *args: Any, expected_status_code: int = 200, **kwargs: Any) -> dict:
    """Call `client.get` with the given arguments and return the JSON content of the response.

    Args:
        client: The TestClient to use for the request.
        *args: Positional arguments to pass to `client.get`.
        expected_status_code: Expected HTTP response status.
        **kwargs: Keyword arguments to pass to `client.get`.

    Returns:
        The JSON content of the response.
    """
    response = client.get(*args, **kwargs)
    assert response.status_code == expected_status_code
    assert "application/json" in response.headers.get("content-type", "")
    path = str(args[0]).partition("?")[0]
    validate_response_contract(client, response, path)
    return response.json()


def make_liststr(arg: str | list[str]) -> str:
    """Return `arg` if it is a string, otherwise return a string joining the items of `arg` with `QUERY_DELIM`."""
    return arg if isinstance(arg, str) else QUERY_DELIM.join(arg)

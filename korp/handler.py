"""API request handler decorator and request processing utilities."""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import inspect
import json
import threading
import time
import traceback
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sized
from dataclasses import dataclass, replace
from functools import update_wrapper
from logging import getLogger
from typing import Any, NoReturn, TypeAlias

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.routing import APIRoute, RouteContext, iter_route_contexts
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from korp.api.requests import CommonQueryControls, QueryRequestModel, RequestModel
from korp.config import settings
from korp.dependencies import AbortSignal, AdminCtxDep, Ctx, CtxDep, QueryCtxDep, build_common_params

logger = getLogger(__name__)

HTTP_INTERNAL_SERVER_ERROR = 500
HTTP_BAD_REQUEST = 400


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """A progress update emitted while producing a streamed response.

    Attributes:
        completed: Number of completed work items.
        total: Total number of work items expected.
        corpus: Corpus whose work item just completed, if applicable.
        corpora: Corpora covered by the operation, included in the initial event.
        hits: Number of hits found for the completed corpus, if known.
    """

    completed: int
    total: int
    corpus: str | None = None
    corpora: list[str] | None = None
    hits: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the public NDJSON event representation."""
        event: dict[str, Any] = {"event": "progress", "completed": self.completed, "total": self.total}
        for field_name in ("corpus", "corpora", "hits"):
            if (value := getattr(self, field_name)) is not None:
                event[field_name] = value
        return event


ResponseFragment: TypeAlias = dict[str, Any] | ProgressEvent


def _reject_unsupported_route_result(result: object) -> NoReturn:
    """Reject a route result that is not of a supported type.

    Raises:
        TypeError: Always, with the unsupported result type.
    """
    raise TypeError(
        "API routes must return a dictionary, an iterator of result fragments, or a Response; "
        f"got {type(result).__name__}."
    )


def _validate_result_fragment(fragment: object, seen_keys: set[str]) -> dict[str, Any]:
    """Validate a result fragment emitted by a route, ensuring it is a non-empty object with unique string keys.

    Args:
        fragment: Next value emitted by the route.
        seen_keys: Top-level keys contributed by preceding fragments.

    Returns:
        The validated result fragment.

    Raises:
        TypeError: If the fragment is not an object or has a non-string key.
        ValueError: If the fragment is empty, contains the reserved `elapsed` field, or repeats a preceding key.
    """
    if not isinstance(fragment, dict):
        raise TypeError(f"Result fragments must be dictionaries, got {type(fragment).__name__}.")
    if not fragment:
        raise ValueError("Result fragments must not be empty.")

    if any(not isinstance(key, str) for key in fragment):
        raise TypeError("Result fragment keys must be strings.")
    if "elapsed" in fragment:
        raise ValueError("The result key 'elapsed' is reserved for the complete event.")

    duplicate_keys = seen_keys & fragment.keys()
    if duplicate_keys:
        names = ", ".join(sorted(duplicate_keys))
        raise ValueError(f"Result fragment repeats top-level key(s): {names}.")

    seen_keys.update(fragment)
    return fragment


def _require_result_fields(seen_keys: Sized) -> None:
    """Require a decorated route to produce at least one result field.

    Raises:
        ValueError: If the route produced no result fragments.
    """
    if not seen_keys:
        raise ValueError("API routes must produce at least one non-empty result object.")


def iter_api_route_contexts(app: FastAPI) -> Iterator[RouteContext]:
    """Iterate over route contexts whose original routes are API routes.

    Yields:
        FastAPI route contexts for ``APIRoute`` instances.
    """
    for context in iter_route_contexts(app.routes):
        if isinstance(context.original_route, APIRoute):
            yield context


class APIValidationError(HTTPException):
    """Signal a request validation failure with the API's standard error representation.

    Use this exception to indicate that the request was invalid, for example incomplete or inconsistent parameters. This
    is distinct from FastAPI's built-in validation errors, which are raised automatically. Raising this exception will
    result in a JSON response with HTTP status 422 and the `invalid_request` error code. The optional `field` value
    identifies the related request field in the error response.

    This can be raised either by a FastAPI dependency or by a route before response processing begins. In both cases,
    the error is formatted into the same public error shape. An exception raised after a streaming response has started
    cannot change the HTTP status and is emitted as a streamed error event instead.
    """

    def __init__(self, detail: str, *, field: str | None = None) -> None:
        """Create a validation error with HTTP status 422 and the given detail message."""
        super().__init__(status_code=422, detail=detail)
        self.field = field

    def __str__(self) -> str:
        """Return the validation detail without the HTTP status prefix."""
        return str(self.detail)


def _unwrap_error(exc: BaseException) -> BaseException:
    """Return the most useful leaf exception from a wrapped error.

    AnyIO task groups can surface failures as `ExceptionGroup` instances, which are too generic to expose directly in
    API responses. Prefer the first non-cancellation leaf with a message, and fall back to the original exception if we
    cannot find a better candidate.

    Args:
        exc: The exception to unwrap.

    Returns:
        The most informative underlying exception.
    """

    def iter_leaves(error: BaseException) -> list[BaseException]:
        if isinstance(error, BaseExceptionGroup):
            leaves: list[BaseException] = []
            for child in error.exceptions:
                leaves.extend(iter_leaves(child))
            return leaves
        return [error]

    leaves = [leaf for leaf in iter_leaves(exc) if not isinstance(leaf, asyncio.CancelledError)]
    if not leaves:
        return exc

    for leaf in leaves:
        if str(leaf):
            return leaf

    return leaves[0]


def docs_error_responses(http_errors: Mapping[int, str] | None = None) -> dict[int | str, dict[str, Any]]:
    """Document errors raised before the response starts.

    Args:
        http_errors: Additional HTTPException status codes mapped to route-specific descriptions, for example
            `{404: "The requested resource was not found."}` or `{429: "Configured rate limit exceeded."}`.
            These declarations describe existing behavior; they do not enable authorization or rate limiting.

    Returns:
        FastAPI response declarations for validation, unexpected server errors, and the supplied HTTP errors.
    """
    from korp.api.schemas import ErrorResponse  # noqa: PLC0415

    responses: dict[int | str, dict[str, Any]] = {
        400: {
            "model": ErrorResponse,
            "description": "The request or CQP query was invalid.",
        },
        422: {
            "model": ErrorResponse,
            "description": "Request validation or preflight processing failed.",
        },
        500: {
            "model": ErrorResponse,
            "description": "Unexpected server error before the response started.",
        },
        503: {
            "model": ErrorResponse,
            "description": "A required backend service was unavailable.",
        },
    }
    for code, description in (http_errors or {}).items():
        responses[code] = {"model": ErrorResponse, "description": description}
    return responses


def docs_response(
    model: type[Any],
    *,
    status_code: int = 200,
    description: str | None = None,
    corpus_authorization: bool = False,
    http_errors: Mapping[int, str] | None = None,
) -> dict[int | str, dict[str, Any]]:
    """Build OpenAPI response documentation without enabling response processing.

    Use together with `response_model=None` on the route decorator to keep docs while avoiding runtime response
    validation/serialization overhead.

    Args:
        model: The response model class to document.
        status_code: The HTTP status code for the documented response.
        description: Optional description for the documented response.
        corpus_authorization: Whether to document the HTTP 403 response from corpus authorization. This only describes
            existing behavior; it does not enable authorization.
        http_errors: Additional pre-response HTTP errors; see `docs_error_responses`.

    Returns:
        A dictionary suitable for the `responses` parameter of FastAPI route decorators.
    """
    from pydantic import TypeAdapter  # noqa: PLC0415

    from korp.api.schemas import StreamEvent  # noqa: PLC0415

    response: dict[str, Any] = {"model": model}

    # Add NDJSON schema for streaming responses.
    # The regular application/json schema is added automatically by FastAPI.
    ndjson_record_schema = TypeAdapter(StreamEvent).json_schema()
    ndjson_record_schema["description"] = (
        "Schema for each line in the NDJSON stream returned when `stream=true`. To assemble a successful response, "
        "start with an empty object and copy each top-level member of every result event's `data` object into it in "
        "the order they appear. A top-level result key may occur in only one result event, so no merging or appending "
        "of nested objects or arrays is needed. A successful stream contains at least one non-empty result event. "
        "After `complete` with `ok=true`, copy its `elapsed` value into the assembled object. The assembled object "
        "then conforms to this operation's ordinary `application/json` success schema."
    )
    response["content"] = {
        "application/x-ndjson": {
            "schema": ndjson_record_schema,
        }
    }
    if description is not None:
        response["description"] = description
    errors = {403: "Access to a requested corpus was denied."} if corpus_authorization else {}
    errors.update(http_errors or {})
    responses = docs_error_responses(errors)
    responses[status_code] = response
    return responses


def forbid_extra_query_params(request: Request) -> None:
    """Raise HTTP 422 if the request contains unexpected query parameters.

    Args:
        request: The FastAPI Request object.

    Raises:
        HTTPException: If unexpected query parameters are found.
    """
    route = request.scope.get("route")
    if not isinstance(route, APIRoute):
        return
    allowed: set[str] = set()
    dependants = [route.dependant]
    visited: set[int] = set()
    while dependants:
        dependant = dependants.pop()
        if id(dependant) in visited:
            continue
        visited.add(id(dependant))
        for parameter in dependant.query_params:
            annotation = parameter.field_info.annotation
            if isinstance(annotation, type) and issubclass(annotation, CommonQueryControls):
                allowed.update(field.alias or name for name, field in annotation.model_fields.items())
            else:
                allowed.add(parameter.alias)
        dependants.extend(dependant.dependencies)
    extra = set(request.query_params) - allowed
    if extra:
        raise HTTPException(422, f"Unexpected query params: {', '.join(sorted(extra))}")


def enforce_ctx_dependency(
    app: FastAPI,
) -> None:
    """Strictly enforce that every APIRoute endpoint has the required 'ctx' parameter.

    Every route is expected to have a parameter named 'ctx' or '_ctx' with the annotation 'CtxDep', 'QueryCtxDep', or
    'AdminCtxDep', which injects the request context, containing common parameters and other commonly used objects. GET
    routes using a query request model need to use 'QueryCtxDep'. Administrative routes without public response controls
    use 'AdminCtxDep'.

    Raises:
        RuntimeError: If any route is missing the required 'ctx' or '_ctx' parameter or has incorrect annotation.
    """
    param_name = "ctx"
    ctx_dependencies = {CtxDep, QueryCtxDep, AdminCtxDep}
    ctx_dependency_name = "CtxDep, QueryCtxDep, or AdminCtxDep"  # For error messages
    violations: list[str] = []

    for route_context in iter_api_route_contexts(app):
        r = route_context.original_route
        assert isinstance(r, APIRoute)
        path = route_context.path_format
        assert path is not None
        signature = inspect.signature(r.endpoint)
        p = signature.parameters.get(param_name) or signature.parameters.get(f"_{param_name}")

        methods = f"[{','.join(sorted(r.methods or []))}]"
        where = f"{path:30s} {methods:12s} {r.endpoint.__module__}.{r.endpoint.__name__}"
        if p is None:
            violations.append(f"{where}\n  - missing required parameter `{param_name}: {ctx_dependency_name}`")
            continue

        if p.annotation not in ctx_dependencies:
            violations.append(
                f"{where}\n  - `{param_name}` requires annotation `{ctx_dependency_name}`, found `{p.annotation}`"
            )
            continue

        # If a route uses QueryCtxDep, it must have exactly one parameter that is a subclass of QueryRequestModel.
        if p.annotation is QueryCtxDep:
            query_models = [
                parameter.field_info.annotation
                for parameter in r.dependant.query_params
                if isinstance(parameter.field_info.annotation, type)
                and issubclass(parameter.field_info.annotation, QueryRequestModel)
            ]
            if len(query_models) != 1:
                violations.append(f"{where}\n  - `QueryCtxDep` requires exactly one `QueryRequestModel` parameter")

    if violations:
        raise RuntimeError("\nCtx dependency check failed.\n\n" + "\n\n".join(violations))


def _problem_details(exc: BaseException, *, debug: bool) -> dict[str, Any]:
    """Convert an exception to the public Problem Details-style error shape.

    Returns:
        The serialized public error object.
    """
    from korp import cqp  # noqa: PLC0415
    from korp.memcached import CacheError  # noqa: PLC0415

    error = _unwrap_error(exc)
    field: str | None = None
    errors: list[dict[str, Any]] | None = None

    if isinstance(error, RequestValidationError):
        code = "invalid_request"
        title = "Invalid request"
        status = 422
        detail = "Request validation failed."
        errors = jsonable_encoder(error.errors())
    elif isinstance(error, APIValidationError):
        code = "invalid_request"
        title = "Invalid request"
        status = error.status_code
        detail = error.detail
        field = error.field
    elif isinstance(error, StarletteHTTPException):
        status = error.status_code
        code, title = {
            400: ("invalid_request", "Invalid request"),
            401: ("authentication_required", "Authentication required"),
            403: ("access_denied", "Access denied"),
            404: ("not_found", "Not found"),
            409: ("conflict", "Conflict"),
            422: ("invalid_request", "Invalid request"),
            429: ("rate_limit_exceeded", "Rate limit exceeded"),
        }.get(status, ("http_error", "HTTP error"))
        detail = error.detail
    elif isinstance(error, cqp.CQPError):
        code = "cqp_error"
        title = "CQP query failed"
        status = 400
        detail = str(error)
    elif isinstance(error, SQLAlchemyError):
        code = "database_unavailable"
        title = "Database unavailable"
        status = 503
        detail = "The database backend is unavailable."
    elif isinstance(error, CacheError):
        code = "cache_unavailable"
        title = "Cache unavailable"
        status = 503
        detail = "The cache backend is unavailable."
    else:
        code = "backend_error"
        title = "Backend error"
        status = 500
        detail = "An unexpected backend error occurred."

    problem: dict[str, Any] = {"code": code, "title": title, "status": status, "detail": detail}
    if field is not None:
        problem["field"] = field
    if errors is not None:
        problem["errors"] = errors
    if settings.ERROR_TRACEBACKS_ENABLED and debug:
        tb = traceback.format_exception(type(error), error, error.__traceback__)
        problem["traceback"] = [line.rstrip("\n") for line in tb]
    return problem


def _problem_response(exc: BaseException, *, debug: bool) -> JSONResponse:
    """Create a JSON error response, preserving headers from HTTP exceptions.

    Returns:
        A response carrying the public error object and its associated status.
    """
    problem = _problem_details(exc, debug=debug)
    error = _unwrap_error(exc)
    headers = error.headers if isinstance(error, StarletteHTTPException) else None
    return JSONResponse(status_code=problem["status"], content=problem, headers=headers)


def _route_disables_cache(request: Request) -> bool:
    """Return whether the matched route opts out of public response caching."""
    route = request.scope.get("route")
    return isinstance(route, APIRoute) and getattr(route.endpoint, "_korp_cache_headers", True) is False


def _request_debug_enabled(request: Request) -> bool:
    """Return whether the request explicitly enabled debug output."""
    return request.query_params.get("debug", "").lower() in {"1", "true", "yes", "on"}


def install_error_handlers(app: FastAPI) -> None:
    """Install the API's unified JSON exception handlers on the FastAPI app."""

    async def request_validation_handler(request: Request, exc: Exception) -> JSONResponse:  # noqa: RUF029
        assert isinstance(exc, RequestValidationError)
        response = _problem_response(exc, debug=_request_debug_enabled(request))
        if _route_disables_cache(request):
            response.headers["Cache-Control"] = "no-store"
        return response

    async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:  # noqa: RUF029
        assert isinstance(exc, StarletteHTTPException)
        response = _problem_response(exc, debug=_request_debug_enabled(request))
        if _route_disables_cache(request):
            response.headers["Cache-Control"] = "no-store"
        return response

    async def unexpected_exception_handler(request: Request, exc: Exception) -> JSONResponse:  # noqa: RUF029
        logger.error(
            "Unhandled API error: %s %s",
            request.method,
            request.url.path,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        response = _problem_response(exc, debug=_request_debug_enabled(request))
        if _route_disables_cache(request):
            response.headers["Cache-Control"] = "no-store"
        return response

    app.add_exception_handler(RequestValidationError, request_validation_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unexpected_exception_handler)


def _set_cache_headers(
    resp: Response,
    *,
    max_age_seconds: int,
    private: bool = False,
    vary_headers: tuple[str, ...] = (),
) -> None:
    """Set HTTP cache headers on the response.

    Args:
        resp: The FastAPI Response object.
        max_age_seconds: The max-age in seconds for the Cache-Control header.
        private: Whether only a private client cache may store the response.
        vary_headers: Request header names that must match for a cached response to be reused.
    """
    expires = datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=max_age_seconds)
    resp.headers["Expires"] = expires.strftime("%a, %d %b %Y %H:%M:%S GMT")
    visibility = "private" if private else "public"
    resp.headers["Cache-Control"] = f"{visibility},max-age={max_age_seconds}"
    if vary_headers:
        existing = [header.strip() for header in resp.headers.get("Vary", "").split(",") if header.strip()]
        existing_lower = {header.lower() for header in existing}
        for header in vary_headers:
            if header.lower() not in existing_lower:
                existing.append(header)
                existing_lower.add(header.lower())
        resp.headers["Vary"] = ", ".join(existing)


def _apply_response_cache_policy(
    resp: Response,
    request: Request,
    *,
    cache_headers: bool,
    cache_requested: bool,
    debug: bool,
    streaming: bool,
) -> None:
    """Apply HTTP response caching policy based on route, request, and server settings."""
    authorizer = getattr(request.app.state, "authorizer", None)

    # If caching is explicitly disabled on the route, or the request is in debug or streaming mode, use `no-store`.
    if not cache_headers or debug or streaming:
        resp.headers["Cache-Control"] = "no-store"
        # A pass-through Response may already contain Expires, which is inconsistent with `no-store`.
        if "Expires" in resp.headers:
            del resp.headers["Expires"]
        return

    # If the response is an error or caching is disabled (either by the client or server-side), leave out cache
    # headers. With an authorizer, mark the response `no-store` to avoid leaking credential-dependent information.
    if resp.status_code >= HTTP_BAD_REQUEST or not cache_requested:
        if authorizer is not None:
            resp.headers["Cache-Control"] = "no-store"
        return

    max_age = settings.HTTP_CACHE_MAXAGE * 3600
    # Interpret a negative configured lifetime as "do not add cache headers" for anonymous responses. For authorized
    # responses, treat it as an explicit storage prohibition.
    if max_age <= 0:
        if authorizer is not None:
            resp.headers["Cache-Control"] = "no-store"
        return

    # Without authorization enabled, we can safely add public cache headers to successful responses.
    if authorizer is None:
        _set_cache_headers(resp, max_age_seconds=max_age)
        return

    # With authorization enabled, responses are only cacheable if the plugin explicitly describes every request header
    # that can affect authorization. In that case, we mark the response as `private` and add a `Vary` header. If the
    # plugin does not declare its credential headers, we mark the response `no-store`.
    cache_vary_headers = getattr(authorizer, "cache_vary_headers", None)
    vary_headers = cache_vary_headers() if cache_vary_headers is not None else None
    if vary_headers is None:
        resp.headers["Cache-Control"] = "no-store"
        return
    _set_cache_headers(resp, max_age_seconds=max_age, private=True, vary_headers=vary_headers)


def api_handler(
    _callable: Callable | None = None,
    *,
    cache_headers: bool = True,
    keepalive_seconds: float = 60.0,
    rate_limit: bool = True,
) -> Callable:
    """Main decorator for API routes.

    This decorator is to be used on all API routes. It provides the following features:

    - It produces a single JSON object normally, or an NDJSON event stream when `common.stream` is enabled.
    - In NDJSON mode, it prevents proxy timeouts by sending typed keepalive events regularly.
    - It handles client disconnects and signals the route to abort processing.
    - It returns ordinary failures with HTTP error statuses and a shared JSON error model.
    - It sets HTTP cache headers if enabled.
    - It adds timing information to the output.
    - It indents JSON output if requested (only for non-streaming responses).

    The decorator handles both synchronous and asynchronous endpoints. Synchronous endpoints are run in a thread to
    avoid blocking the event loop. Async endpoints run in the event loop as usual.

    Output and keepalive behavior:
    - Async generators and sync generators, whether yielded directly by the route or returned by it, are consumed before
      an ordinary JSON response is created. With `stream=true`, they stream results and enable keepalive events.
    - A regular route coroutine can perform preflight validation and setup, then return an async or sync iterator for
      the long-running work. Work done before that iterator is returned cannot send NDJSON keepalive events, so
      long-running streamed work should happen in the returned iterator.
    - A route can also return a single dictionary. It is sent as one result, and keepalive output cannot be sent while
      the route computes it.
    - Every successful route must produce at least one non-empty result object. The `elapsed` field is response metadata
      added by the handler and is not sufficient result data by itself.

    Every route is required to have the following parameter (named either "ctx" or "_ctx"), which injects the request
    context, containing common parameters and other commonly used objects. GET routes using a query request model
    (usually routes with corresponding POST routes) use `QueryCtxDep`; administrative routes without public response
    controls use `AdminCtxDep`; all other routes use `CtxDep`:

        ctx: CtxDep

    An optional parameter may also be declared for routes that need abort signaling:

        abort_signal: AbortSignal = None

    To check if abort is requested, use `abort_signal.is_set()`.

    Decorated routes can either:
      - yield dict result fragments or ProgressEvent objects from a generator or async generator
      - return a sync or async iterator yielding dict result fragments or ProgressEvent objects
      - return a dict to send as one result
      - return Response to bypass the decorator's JSON streaming, error formatting, and timing output. Cache headers
        and rate-limit headers may still be added.

    The decorator can be used with or without parentheses:
        @api_handler
        @api_handler(cache_headers=False)
        @api_handler(rate_limit=False)

    Args:
        _callable: The route function to decorate.
        cache_headers: Whether successful responses may receive public cache headers. When false, responses use
            `Cache-Control: no-store`. With an authorizer enabled, ordinary responses use private browser caching when
            the plugin declares its credential headers; otherwise they use `no-store`. Debug and streamed responses are
            never stored.
        keepalive_seconds: Interval in seconds for sending NDJSON keepalive events.
        rate_limit: Whether this route is eligible for rate limiting (default `True`). Set to `False` to exempt it.
            The actual limit values are read from configuration (`RATE_LIMIT_DEFAULT` and `RATE_LIMITS`). Has no effect
            unless the global rate-limiter is enabled and at least one limit is configured.

    Returns:
        The decorated route function.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Awaitable[Response]]:
        uses_query_ctx = any(
            parameter.annotation is QueryCtxDep for parameter in inspect.signature(fn).parameters.values()
        )
        expects_json_body = any(
            isinstance(parameter.annotation, type) and issubclass(parameter.annotation, RequestModel)
            for parameter in inspect.signature(fn).parameters.values()
        )

        async def wrapper(*args: Any, **kwargs: Any) -> Response:
            ctx: Ctx = kwargs.get("ctx") or kwargs["_ctx"]  # Support both "ctx" and "_ctx"
            request = ctx.request
            if uses_query_ctx:
                query = next((value for value in kwargs.values() if isinstance(value, QueryRequestModel)), None)
                if query is None:
                    raise RuntimeError("QueryCtxDep requires a QueryRequestModel parameter")
                # Build the common controls from the query model and replace the placeholder in the context
                ctx = replace(ctx, common=build_common_params(request, query))
                kwargs["ctx" if "ctx" in kwargs else "_ctx"] = ctx
            common = ctx.common
            route = request.url.path
            method = request.method

            if expects_json_body and method == "POST":
                content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if content_type != "application/json":
                    raise HTTPException(422, "POST request bodies must use application/json.")

            # Check for unexpected query parameters
            forbid_extra_query_params(request)

            rate_limit_headers: dict[str, str] = {}
            if rate_limit and (app_rate_limiter := getattr(request.app.state, "rate_limiter", None)):
                from korp.rate_limit import resolve_rate_limit  # noqa: PLC0415

                effective_limit = resolve_rate_limit(route, settings=settings)
                if effective_limit is not None:
                    check = await app_rate_limiter.check_request(request, limit=effective_limit)
                    rate_limit_headers = check.headers
                    if not check.allowed:
                        raise HTTPException(
                            status_code=429,
                            detail="Rate limit exceeded.",
                            headers=rate_limit_headers or None,
                        )

            abort = AbortSignal(threading.Event(), asyncio.Event(), asyncio.get_running_loop())

            # Inject abort signal if endpoint accepts it
            if "abort_signal" in inspect.signature(fn).parameters:
                kwargs["abort_signal"] = abort

            start = time.perf_counter()
            slow_request_threshold = max(0.0, settings.REQUEST_SLOW_LOG_SECONDS)
            stuck_log_interval = max(1.0, settings.REQUEST_STUCK_LOG_INTERVAL_SECONDS)
            watchdog_task: asyncio.Task[None] | None = None

            if slow_request_threshold > 0:

                async def request_watchdog() -> None:
                    await asyncio.sleep(slow_request_threshold)
                    while True:
                        elapsed = time.perf_counter() - start
                        logger.warning("Request still running %.3fs: %s %s", elapsed, method, route)
                        await asyncio.sleep(stuck_log_interval)

                watchdog_task = asyncio.create_task(request_watchdog())

            async def stop_watchdog() -> None:
                if watchdog_task is None:
                    return
                watchdog_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watchdog_task

            # Call route. Generator routes will return generator objects.
            try:
                if inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn):
                    result = fn(*args, **kwargs)
                    if inspect.iscoroutine(result):
                        result = await result
                else:
                    result = await asyncio.to_thread(fn, *args, **kwargs)
            except asyncio.CancelledError:
                abort.set()
                await stop_watchdog()
                raise
            except Exception as exc:
                elapsed = time.perf_counter() - start
                if slow_request_threshold > 0 and elapsed >= slow_request_threshold:
                    logger.warning("Slow request %.3fs: %s %s", elapsed, method, route)
                await stop_watchdog()
                response = _problem_response(exc, debug=common.debug)
                _apply_response_cache_policy(
                    response,
                    request,
                    cache_headers=cache_headers,
                    cache_requested=common.cache,
                    debug=common.debug,
                    streaming=common.stream,
                )
                if response.status_code >= HTTP_INTERNAL_SERVER_ERROR:
                    logger.exception("API route failed before producing a result: %s %s", method, route, exc_info=exc)
                for header_name, header_value in rate_limit_headers.items():
                    response.headers[header_name] = header_value
                return response

            # Pass-through if Response explicitly returned
            if isinstance(result, Response):
                elapsed = time.perf_counter() - start
                if slow_request_threshold > 0 and elapsed >= slow_request_threshold:
                    logger.warning("Slow request %.3fs: %s %s", elapsed, method, route)
                await stop_watchdog()
                _apply_response_cache_policy(
                    result,
                    request,
                    cache_headers=cache_headers,
                    cache_requested=common.cache,
                    debug=common.debug,
                    streaming=common.stream,
                )
                for header_name, header_value in rate_limit_headers.items():
                    result.headers[header_name] = header_value
                return result

            fragments = result  # dict or iterator/generator

            if not common.stream:
                result_obj: dict[str, Any] = {}
                seen_result_keys: set[str] = set()

                def merge_fragment(fragment: object) -> None:
                    result_obj.update(_validate_result_fragment(fragment, seen_result_keys))

                try:
                    if isinstance(fragments, dict):
                        merge_fragment(fragments)
                    elif hasattr(fragments, "__aiter__"):
                        async for item in fragments:
                            if isinstance(item, ProgressEvent):
                                continue
                            merge_fragment(item)
                    elif hasattr(fragments, "__iter__") and not isinstance(
                        fragments, (str, bytes, bytearray, list, tuple)
                    ):

                        def consume_sync_iter() -> dict[str, Any]:
                            merged: dict[str, Any] = {}
                            seen_keys: set[str] = set()
                            for item in fragments:
                                if abort.is_set():
                                    break
                                if isinstance(item, ProgressEvent):
                                    continue
                                merged.update(_validate_result_fragment(item, seen_keys))
                            return merged

                        # Run the synchronous iterator in a thread to avoid blocking the event loop
                        result_obj.update(await asyncio.to_thread(consume_sync_iter))
                    else:
                        _reject_unsupported_route_result(fragments)
                    _require_result_fields(result_obj)
                except asyncio.CancelledError:
                    abort.set()
                    await stop_watchdog()
                    raise
                except Exception as exc:
                    response = _problem_response(exc, debug=common.debug)
                    if response.status_code >= HTTP_INTERNAL_SERVER_ERROR:
                        logger.exception("API route failed: %s %s", method, route, exc_info=exc)
                else:
                    result_obj["elapsed"] = time.perf_counter() - start
                    if common.indent > 0:
                        response = Response(
                            content=json.dumps(result_obj, indent=common.indent), media_type="application/json"
                        )
                    else:
                        response = JSONResponse(content=result_obj)

                elapsed = time.perf_counter() - start
                if slow_request_threshold > 0 and elapsed >= slow_request_threshold:
                    logger.warning("Slow request %.3fs: %s %s", elapsed, method, route)
                await stop_watchdog()
                _apply_response_cache_policy(
                    response,
                    request,
                    cache_headers=cache_headers,
                    cache_requested=common.cache,
                    debug=common.debug,
                    streaming=common.stream,
                )
                for header_name, header_value in rate_limit_headers.items():
                    response.headers[header_name] = header_value
                return response

            queue: asyncio.Queue[Any] = asyncio.Queue()

            async def producer() -> None:
                """Push fragments into asyncio queue.

                Sync generators are run in a thread.

                Producer also respects abort_signal to stop processing if client disconnects.
                """
                loop = asyncio.get_running_loop()
                seen_result_keys: set[str] = set()

                def validate_item(item: object) -> None:
                    if not isinstance(item, ProgressEvent):
                        _validate_result_fragment(item, seen_result_keys)

                try:
                    # Result is dict -> push and exit
                    if isinstance(fragments, dict):
                        validate_item(fragments)
                        await queue.put(fragments)
                        return

                    # Async generator/iterator -> iterate
                    if hasattr(fragments, "__aiter__"):
                        async for item in fragments:
                            if abort.is_set():
                                return
                            validate_item(item)
                            await queue.put(item)
                        _require_result_fields(seen_result_keys)
                        return

                    # Sync generator/iterator -> run in thread
                    if hasattr(fragments, "__iter__") and not isinstance(
                        fragments, (str, bytes, bytearray, list, tuple)
                    ):

                        def run_sync_iter() -> None:
                            for item in fragments:
                                if abort.is_set():
                                    return
                                validate_item(item)
                                loop.call_soon_threadsafe(queue.put_nowait, item)

                        await asyncio.to_thread(run_sync_iter)
                        if abort.is_set():
                            return
                        _require_result_fields(seen_result_keys)
                        return

                    # Any other return type is unsupported and rejected
                    _reject_unsupported_route_result(fragments)

                except Exception as exc:
                    await queue.put(exc)
                finally:
                    await queue.put(None)  # Sentinel to indicate end of stream

            async def body_iter_stream() -> AsyncIterator[bytes]:
                producer_task = asyncio.create_task(producer())
                finished = False
                ok = True
                try:
                    while True:
                        try:
                            item = await asyncio.wait_for(queue.get(), timeout=keepalive_seconds)
                        except TimeoutError:
                            # Check disconnect only when idle to avoid false positives
                            if await request.is_disconnected():
                                abort.set()
                                return
                            yield b'{"event": "keepalive"}\n'
                            continue

                        if item is None:
                            # End of stream
                            break

                        if isinstance(item, Exception):
                            ok = False
                            problem = _problem_details(item, debug=common.debug)
                            if problem["status"] >= HTTP_INTERNAL_SERVER_ERROR:
                                logger.error(
                                    "API stream failed: %s %s",
                                    method,
                                    route,
                                    exc_info=(type(item), item, item.__traceback__),
                                )
                            yield (json.dumps({"event": "error", "error": problem}) + "\n").encode("utf-8")
                            break

                        event = item.as_dict() if isinstance(item, ProgressEvent) else {"event": "result", "data": item}
                        yield (json.dumps(event) + "\n").encode("utf-8")

                    yield (
                        json.dumps({"event": "complete", "ok": ok, "elapsed": time.perf_counter() - start}) + "\n"
                    ).encode("utf-8")
                    finished = True

                except asyncio.CancelledError:
                    # Client disconnected
                    abort.set()
                    raise
                finally:
                    if not finished:
                        # If we're exiting for non-cancel reasons, also set abort to stop work
                        abort.set()
                    producer_task.cancel()
                    elapsed = time.perf_counter() - start
                    if slow_request_threshold > 0 and elapsed >= slow_request_threshold:
                        logger.warning("Slow request %.3fs: %s %s", elapsed, method, route)
                    await stop_watchdog()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await producer_task

            resp = StreamingResponse(body_iter_stream(), media_type="application/x-ndjson")

            _apply_response_cache_policy(
                resp,
                request,
                cache_headers=cache_headers,
                cache_requested=common.cache,
                debug=common.debug,
                streaming=common.stream,
            )
            for header_name, header_value in rate_limit_headers.items():
                resp.headers[header_name] = header_value

            return resp

        # FastAPI 0.134+ unwraps decorated callables to detect generator endpoints. Our wrapper is always an async
        # coroutine, even when `fn` is an async generator. Exposing `__wrapped__` makes FastAPI classify this as an
        # async-generator route, which then crashes when it tries to `async for` over a coroutine. We can work around
        # this by deleting `__wrapped__` after updating the wrapper to look like `fn`, instead of using
        # `functools.wraps`.
        update_wrapper(wrapper, fn)
        vars(wrapper)["__signature__"] = inspect.signature(fn)
        vars(wrapper)["_korp_rate_limit"] = rate_limit
        vars(wrapper)["_korp_cache_headers"] = cache_headers
        vars(wrapper).pop("__wrapped__", None)

        return wrapper

    # If called as @api_handler without parentheses
    if _callable is not None:
        return decorator(_callable)

    # If called as @api_handler(...) with parentheses
    return decorator

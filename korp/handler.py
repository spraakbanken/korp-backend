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
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from functools import update_wrapper
from logging import getLogger
from typing import Any, TypeAlias
from urllib.parse import parse_qsl, urlencode

from fastapi import FastAPI, HTTPException, Request
from fastapi.dependencies.utils import get_flat_dependant
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.routing import APIRoute
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from korp.config import settings
from korp.dependencies import AbortSignal, Ctx, CtxDep

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


class APIValidationError(HTTPException):
    """Report invalid request input before the response body begins.

    Raise this exception from a FastAPI dependency when validation must happen before the route callable is entered,
    for example when validating a relationship between multiple query parameters. FastAPI resolves dependencies before
    invoking the route and recognizes this exception as an intentional HTTP error, returning status 422 instead of
    treating the validation failure as an unhandled server exception.

    It may also be raised directly by a route during its initial setup. In that case, `api_handler` catches it before
    creating the streaming response and returns a formatted 422 response. Errors raised by dependencies use FastAPI's
    standard HTTP-exception response body instead. Validation that happens after the response has started cannot change
    the HTTP status and is handled as a streamed error instead.

    Example:
        def validate_dates(date_from: str | None, date_to: str | None) -> None:
            if date_from and date_to and date_from > date_to:
                raise APIValidationError("date_from must be before or equal to date_to.")
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

    AnyIO task groups can surface failures as ``ExceptionGroup`` instances, which are too generic to expose directly in
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


def docs_error_responses(http_errors: dict[int, str] | None = None) -> dict[int | str, dict[str, Any]]:
    """Document errors raised before the response starts.

    Args:
        http_errors: Additional HTTPException status codes mapped to route-specific descriptions, for example
            `{403: "Access to a requested corpus was denied."}` or `{429: "Configured rate limit exceeded."}`.
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
    http_errors: dict[int, str] | None = None,
) -> dict[int | str, dict[str, Any]]:
    """Build OpenAPI response documentation without enabling response processing.

    Use together with `response_model=None` on the route decorator to keep docs while avoiding runtime response
    validation/serialization overhead.

    Args:
        model: The response model class to document.
        status_code: The HTTP status code for the documented response.
        description: Optional description for the documented response.
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
        "Schema for each line in the NDJSON stream returned when `stream=true`. Result `data` objects are "
        "fragments of the ordinary JSON response and are merged in the order they are received."
    )
    response["content"] = {
        "application/x-ndjson": {
            "schema": ndjson_record_schema,
        }
    }
    if description is not None:
        response["description"] = description
    responses = docs_error_responses(http_errors)
    responses[status_code] = response
    return responses


def _to_query_value(value: Any) -> str:
    """Convert JSON scalar values to query-string values.

    Args:
        value: The value to convert.

    Returns:
        The string representation of the value for use in query parameters.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=True)
    return str(value)


async def convert_post_body_to_query_params(request: Request) -> None:
    """For POST body params, copy fields into query params.

    This modifies the request in-place.

    Supported content types:
    - application/json (top-level JSON object)
    - application/x-www-form-urlencoded

    Existing query params take precedence over body fields.

    Args:
        request: The FastAPI Request object.
    """
    if request.method != "POST":
        return

    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in {"application/json", "application/x-www-form-urlencoded"}:
        return

    if not (body := await request.body()):
        return

    pairs = parse_qsl(request.scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True)
    existing_keys = {key for key, _ in pairs}
    body_pairs: list[tuple[str, str]] = []

    if content_type == "application/json":
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        for key, raw_value in payload.items():
            if isinstance(raw_value, list):
                body_pairs.extend((key, _to_query_value(item)) for item in raw_value)
            else:
                body_pairs.append((key, _to_query_value(raw_value)))
    else:  # application/x-www-form-urlencoded
        body_pairs = parse_qsl(body.decode("utf-8", errors="replace"), keep_blank_values=True)

    for key, value in body_pairs:
        if key in existing_keys:
            continue
        pairs.append((key, value))

    request.scope["query_string"] = urlencode(pairs, doseq=True).encode("latin-1")


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
    flat = get_flat_dependant(route.dependant, skip_repeats=True)
    allowed = {p.alias for p in flat.query_params}
    extra = set(request.query_params) - allowed
    if extra:
        raise HTTPException(422, f"Unexpected query params: {', '.join(sorted(extra))}")


def enforce_ctx_dependency(
    app: FastAPI,
) -> None:
    """Strictly enforce that every APIRoute endpoint has the required 'ctx' parameter.

    Every route is expected to have a parameter named 'ctx' or '_ctx' with the annotation 'CtxDep', which injects
    the request context, containing common parameters and other commonly used objects.

    Raises:
        RuntimeError: If any route is missing the required 'ctx' parameter or has incorrect annotation.
    """
    param_name = "ctx"
    ctx_dependency = CtxDep
    ctx_dependency_name = "CtxDep"  # For error messages
    violations: list[str] = []

    for r in app.routes:
        if not isinstance(r, APIRoute):
            continue

        signature = inspect.signature(r.endpoint)
        p = signature.parameters.get(param_name) or signature.parameters.get(f"_{param_name}")

        methods = f"[{','.join(sorted(r.methods or []))}]"
        where = f"{r.path:30s} {methods:12s} {r.endpoint.__module__}.{r.endpoint.__name__}"
        if p is None:
            violations.append(f"{where}\n  - missing required parameter `{param_name}: {ctx_dependency_name}`")
            continue

        if p.annotation is not ctx_dependency:
            violations.append(
                f"{where}\n  - `{param_name}` requires annotation `{ctx_dependency_name}`, found `{p.annotation}`"
            )
            continue

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


def _request_debug_enabled(request: Request) -> bool:
    """Return whether the request explicitly enabled debug output."""
    return request.query_params.get("debug", "").lower() in {"1", "true", "yes", "on"}


def install_error_handlers(app: FastAPI) -> None:
    """Install the API's unified JSON exception handlers on the FastAPI app."""

    async def request_validation_handler(request: Request, exc: Exception) -> JSONResponse:  # noqa: RUF029
        assert isinstance(exc, RequestValidationError)
        return _problem_response(exc, debug=_request_debug_enabled(request))

    async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:  # noqa: RUF029
        assert isinstance(exc, StarletteHTTPException)
        return _problem_response(exc, debug=_request_debug_enabled(request))

    async def unexpected_exception_handler(request: Request, exc: Exception) -> JSONResponse:  # noqa: RUF029
        logger.error(
            "Unhandled API error: %s %s",
            request.method,
            request.url.path,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return _problem_response(exc, debug=_request_debug_enabled(request))

    app.add_exception_handler(RequestValidationError, request_validation_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unexpected_exception_handler)


def _set_cache_headers(resp: Response, *, max_age_seconds: int) -> None:
    """Set HTTP cache headers on the response.

    Args:
        resp: The FastAPI Response object.
        max_age_seconds: The max-age in seconds for the Cache-Control header.
    """
    expires = datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=max_age_seconds)
    resp.headers["Expires"] = expires.strftime("%a, %d %b %Y %H:%M:%S GMT")
    resp.headers["Cache-Control"] = f"public,max-age={max_age_seconds}"


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
    - A route can also return a single dictionary or another non-iterator value. Such a value is sent as one result, and
      keepalive output cannot be sent while the route computes it.

    Every route is required to have the following parameter (named either "ctx" or "_ctx"), which injects the request
    context, containing common parameters and other commonly used objects:

        ctx: CtxDep

    An optional parameter may also be declared for routes that need abort signaling:

        abort_signal: AbortSignal = None

    To check if abort is requested, use `abort_signal.is_set()`.

    Decorated routes can either:
      - yield dict result fragments or ProgressEvent objects from a generator or async generator
      - return a sync or async iterator yielding dict result fragments or ProgressEvent objects
      - return a dict or another value to send as one result
      - return Response to bypass the decorator's JSON streaming, error formatting, and timing output. Cache headers
        and rate-limit headers may still be added.

    The decorator can be used with or without parentheses:
        @api_handler
        @api_handler(cache_headers=False)
        @api_handler(rate_limit=False)

    Args:
        _callable: The route function to decorate.
        cache_headers: Whether to set HTTP cache headers on the response.
        keepalive_seconds: Interval in seconds for sending NDJSON keepalive events.
        rate_limit: Whether this route is eligible for rate limiting (default `True`). Set to `False` to exempt it.
            The actual limit values are read from configuration (`RATE_LIMIT_DEFAULT` and `RATE_LIMITS`). Has no effect
            unless the global rate-limiter is enabled and at least one limit is configured.

    Returns:
        The decorated route function.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Awaitable[Response]]:
        async def wrapper(*args: Any, **kwargs: Any) -> Response:
            ctx: Ctx = kwargs.get("ctx") or kwargs["_ctx"]  # Support both "ctx" and "_ctx"
            request = ctx.request
            common = ctx.common
            route = request.url.path
            method = request.method

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
                if result.status_code < HTTP_BAD_REQUEST and cache_headers and common.cache and not common.debug:
                    max_age = settings.HTTP_CACHE_MAXAGE * 3600
                    if max_age > 0:
                        _set_cache_headers(result, max_age_seconds=max_age)
                for header_name, header_value in rate_limit_headers.items():
                    result.headers[header_name] = header_value
                return result

            fragments = result  # dict OR iterator/generator OR other value

            if not common.stream:
                result_obj: dict[str, Any] = {}
                try:
                    if isinstance(fragments, dict):
                        result_obj.update(fragments)
                    elif hasattr(fragments, "__aiter__"):
                        async for item in fragments:
                            if item and isinstance(item, dict):
                                result_obj.update(item)
                    elif hasattr(fragments, "__iter__") and not isinstance(
                        fragments, (str, bytes, bytearray, list, tuple)
                    ):

                        def consume_sync_iter() -> dict[str, Any]:
                            merged: dict[str, Any] = {}
                            for item in fragments:
                                if abort.is_set():
                                    break
                                if item and isinstance(item, dict):
                                    merged.update(item)
                            return merged

                        # Run the synchronous iterator in a thread to avoid blocking the event loop
                        result_obj.update(await asyncio.to_thread(consume_sync_iter))
                    else:
                        result_obj["data"] = fragments
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
                if response.status_code < HTTP_BAD_REQUEST and cache_headers and common.cache and not common.debug:
                    max_age = settings.HTTP_CACHE_MAXAGE * 3600
                    if max_age > 0:
                        _set_cache_headers(response, max_age_seconds=max_age)
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
                try:
                    # Result is dict -> push and exit
                    if isinstance(fragments, dict):
                        await queue.put(fragments)
                        return

                    # Async generator/iterator -> iterate
                    if hasattr(fragments, "__aiter__"):
                        async for item in fragments:
                            if abort.is_set():
                                return
                            await queue.put(item)
                        return

                    # Sync generator/iterator -> run in thread
                    if hasattr(fragments, "__iter__") and not isinstance(
                        fragments, (str, bytes, bytearray, list, tuple)
                    ):

                        def run_sync_iter() -> None:
                            for item in fragments:
                                if abort.is_set():
                                    return
                                loop.call_soon_threadsafe(queue.put_nowait, item)

                        await asyncio.to_thread(run_sync_iter)
                        return

                    # Any other return value -> wrap
                    await queue.put({"data": fragments})

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

                        if not item:
                            # Allow routes to yield empty items as keepalive.
                            yield b'{"event": "keepalive"}\n'
                            continue

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

            if cache_headers and common.cache and not common.debug:
                max_age = settings.HTTP_CACHE_MAXAGE * 3600
                if max_age > 0:
                    _set_cache_headers(resp, max_age_seconds=max_age)
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
        vars(wrapper).pop("__wrapped__", None)

        return wrapper

    # If called as @api_handler without parentheses
    if _callable is not None:
        return decorator(_callable)

    # If called as @api_handler(...) with parentheses
    return decorator

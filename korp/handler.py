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
from functools import update_wrapper
from logging import getLogger
from typing import Any
from urllib.parse import parse_qsl, urlencode

from fastapi import FastAPI, HTTPException, Request
from fastapi.dependencies.utils import get_flat_dependant
from fastapi.responses import Response, StreamingResponse
from fastapi.routing import APIRoute

from korp.config import settings
from korp.dependencies import AbortSignal, Ctx, CtxDep

logger = getLogger(__name__)


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


def docs_response(
    model: type[Any],
    *,
    status_code: int = 200,
    description: str | None = None,
) -> dict[int | str, dict[str, Any]]:
    """Build OpenAPI response documentation without enabling response processing.

    Use together with `response_model=None` on the route decorator to keep docs while avoiding runtime response
    validation/serialization overhead.

    Args:
        model: The response model class to document.
        status_code: The HTTP status code for the documented response.
        description: Optional description for the documented response.

    Returns:
        A dictionary suitable for the `responses` parameter of FastAPI route decorators.
    """
    response: dict[str, Any] = {"model": model}
    if description is not None:
        response["description"] = description
    return {status_code: response}


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


def _format_error(exc: BaseException, *, debug: bool) -> dict[str, Any]:
    """Format an error response dictionary.

    Returns:
        A dictionary representing the error response.
    """
    error = _unwrap_error(exc)
    err: dict[str, Any] = {"error": {"type": type(error).__name__, "value": str(error)}}
    if debug:
        tb = traceback.format_exception(type(error), error, error.__traceback__)
        err["error"]["traceback"] = [line.rstrip("\n") for line in tb]
    return err


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
    rate_limit: bool = False,
) -> Callable:
    """Main decorator for API routes.

    This decorator is to be used on all API routes. It provides the following features:

    - It produces JSON output, either incrementally or as a whole, depending on common.incremental. Either way, the
      output is a single JSON object.
    - It prevents proxy timeouts by sending keepalive whitespace regularly.
    - It handles client disconnects and signals the route to abort processing.
    - It formats error responses, including optional tracebacks in debug mode.
    - It sets HTTP cache headers if enabled.
    - It adds timing information to the output.
    - It indents JSON output if requested (only for non-incremental responses).

    The decorator handles both synchronous and asynchronous endpoints. Synchronous endpoints are run in a thread to
    avoid blocking the event loop. Async endpoints run in the event loop as usual.

    Output and keepalive behavior:
    - Async generators and sync generators stream results and enable keepalive output.
    - Non-generator endpoints (sync or async) return a single dict; keepalive cannot be sent while they compute.
      Long-running endpoints should therefore be generators to avoid proxy timeouts.

    Every route is required to have the following parameter (named either "ctx" or "_ctx"), which injects the request
    context, containing common parameters and other commonly used objects:

        ctx: CtxDep

    An optional parameter may also be declared for routes that need abort signaling:

        abort_signal: AbortSignal = None

    To check if abort is requested, use `abort_signal.is_set()`.

    Decorated routes can either:
      - yield dict fragments for incremental output
      - return dict
      - return Response (bypasses decorator processing)

    The decorator can be used with or without parentheses:
        @api_handler
        @api_handler(cache_headers=False)
        @api_handler(rate_limit=True)

    Args:
        _callable: The route function to decorate.
        cache_headers: Whether to set HTTP cache headers on the response.
        keepalive_seconds: Interval in seconds for sending keepalive whitespace.
        rate_limit: Mark this route as a rate-limit candidate. The actual limit values are read from configuration
            (`RATE_LIMIT_DEFAULT` and `RATE_LIMITS`). Has no effect unless the global rate-limiter is enabled
            and at least one limit is configured.

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
            except BaseException:
                elapsed = time.perf_counter() - start
                if slow_request_threshold > 0 and elapsed >= slow_request_threshold:
                    logger.warning("Slow request %.3fs: %s %s", elapsed, method, route)
                await stop_watchdog()
                raise

            # Pass-through if Response explicitly returned
            if isinstance(result, Response):
                elapsed = time.perf_counter() - start
                if slow_request_threshold > 0 and elapsed >= slow_request_threshold:
                    logger.warning("Slow request %.3fs: %s %s", elapsed, method, route)
                await stop_watchdog()
                if cache_headers and common.cache and not common.debug:
                    max_age = settings.HTTP_CACHE_MAXAGE * 3600
                    if max_age > 0:
                        _set_cache_headers(result, max_age_seconds=max_age)
                for header_name, header_value in rate_limit_headers.items():
                    result.headers[header_name] = header_value
                return result

            fragments = result  # dict OR iterator/generator OR other value

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

            async def body_iter_incremental() -> AsyncIterator[bytes]:
                producer_task = asyncio.create_task(producer())
                finished = False
                try:
                    yield b"{\n"

                    while True:
                        try:
                            item = await asyncio.wait_for(queue.get(), timeout=keepalive_seconds)
                        except TimeoutError:
                            # Check disconnect only when idle to avoid false positives
                            if await request.is_disconnected():
                                abort.set()
                                return
                            # Send keepalive whitespace to keep connection open
                            yield b" \n"
                            continue

                        if item is None:
                            # End of stream
                            break

                        if isinstance(item, Exception):
                            err = _format_error(item, debug=common.debug)
                            yield (json.dumps(err)[1:-1] + ",\n").encode("utf-8")
                            break

                        if not item:
                            # Allow routes to yield empty items as keepalive
                            yield b" \n"
                            continue

                        yield (json.dumps(item)[1:-1] + ",\n").encode("utf-8")

                    # Always close JSON for connected clients
                    yield (json.dumps({"elapsed": time.perf_counter() - start})[1:] + "\n").encode("utf-8")
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

            keepalive = object()

            # We need a separate keepalive ticker task for non-incremental responses, since we can't rely on queue
            # timeouts in that case (the route might still be producing output incrementally, but we won't be sending it
            # until the end)
            async def ticker() -> None:
                try:
                    while True:
                        await asyncio.sleep(keepalive_seconds)
                        await queue.put(keepalive)
                except asyncio.CancelledError:
                    pass

            async def body_iter_full() -> AsyncIterator[bytes]:
                producer_task = asyncio.create_task(producer())
                ticker_task = asyncio.create_task(ticker())
                result_obj: dict[str, Any] = {}
                finished = False
                try:
                    while True:
                        item = await queue.get()

                        if item is None:
                            break

                        if item is keepalive:
                            if await request.is_disconnected():
                                abort.set()
                                return
                            yield b" \n"
                            continue

                        if isinstance(item, Exception):
                            result_obj = _format_error(item, debug=common.debug)
                            break

                        if not item:
                            yield b" \n"
                            continue

                        if isinstance(item, dict):
                            result_obj.update(item)

                    result_obj["elapsed"] = time.perf_counter() - start
                    indent = common.indent if common.indent > 0 else None
                    yield json.dumps(result_obj, indent=indent).encode("utf-8")
                    finished = True

                except asyncio.CancelledError:
                    abort.set()
                    raise
                finally:
                    if not finished:
                        abort.set()
                    producer_task.cancel()
                    ticker_task.cancel()
                    elapsed = time.perf_counter() - start
                    if slow_request_threshold > 0 and elapsed >= slow_request_threshold:
                        logger.warning("Slow request %.3fs: %s %s", elapsed, method, route)
                    await stop_watchdog()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await producer_task
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await ticker_task

            stream = body_iter_incremental() if common.incremental else body_iter_full()
            resp = StreamingResponse(stream, media_type="application/json")

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
        vars(wrapper).pop("__wrapped__", None)

        return wrapper

    # If called as @api_handler without parentheses
    if _callable is not None:
        return decorator(_callable)

    # If called as @api_handler(...) with parentheses
    return decorator

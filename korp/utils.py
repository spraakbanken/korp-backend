"""Utility functions and classes for the Korp API."""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import hashlib
import inspect
import json
import random
import re
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Generator, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from logging import getLogger
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Annotated, Any, overload
from urllib.parse import parse_qsl, urlencode

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.dependencies.utils import get_flat_dependant
from fastapi.responses import Response, StreamingResponse
from fastapi.routing import APIRoute

if TYPE_CHECKING:
    from korp.cwb import CWB
    from korp.db import MySQL
    from korp.memcached import Memcached, MemcachedSyncClient

from korp.db import escape_string as _db_escape_string

from .config import settings

logger = getLogger(__name__)

# Special symbols used when parsing CQP results; should not appear in corpus data
END_OF_LINE = "-::-EOL-::-"
LEFT_DELIM = "---:::"
RIGHT_DELIM = ":::---"

QUERY_DELIM = ","
UNDEF_VALUE = "__UNDEF__"


def common_params(
    request: Request,
    cache: Annotated[bool, Query(description="Whether to use caching for the request.")] = True,
    debug: Annotated[bool, Query(description="Whether to include debug information in responses.")] = False,
    indent: Annotated[int, Query(ge=0, le=16, description="Number of spaces to indent JSON output.")] = 0,
    incremental: Annotated[
        bool,
        Query(
            description="Whether to produce incremental JSON output. Not every route supports incremental output. "
            "For some routes, you will see partial results as they are generated, while other may only give progress "
            "updates before delivering the final result."
        ),
    ] = False,
) -> CommonParams:
    """FastAPI dependency for common parameters, available in all routes through the Ctx object.

    Args:
        request: The FastAPI Request object.
        cache: Whether to use caching for the request.
        debug: Whether to include debug information in responses.
        indent: Number of spaces to indent JSON output.
        incremental: Whether to produce incremental JSON output.

    Returns:
        A CommonParams object with the parsed parameters.
    """
    effective_cache = request.app.state.cache_enabled and cache

    return CommonParams(
        debug=debug,
        indent=indent,
        incremental=incremental,
        cache=effective_cache,
    )


@dataclass
class CommonParams:
    """Common parameters for API routes.

    Attributes:
        debug: Whether to include debug information in responses.
        indent: Number of spaces to indent JSON output.
        incremental: Whether to produce incremental JSON output.
        cache: Whether to use caching for the request.
    """

    debug: bool = False
    indent: int = 0
    incremental: bool = False
    cache: bool = True


@dataclass(frozen=True)
class Ctx:
    """Context object passed to API routes.

    This object contains commonly used objects for API routes, such as the request, common parameters, and cache client.

    Attributes:
        request: The FastAPI Request object.
        common: The CommonParams for the request.
        cache: The Memcached client.
        db: The database helper.
        cwb: The CWB instance.
    """

    request: Request
    common: CommonParams
    cache: Memcached
    db: MySQL
    cwb: CWB


@dataclass(frozen=True)
class AuthContext:
    """Context object passed to authorizer checks."""

    request: Request
    cache_enabled: bool


def get_ctx(
    request: Request,
    common: Annotated[CommonParams, Depends(common_params)],
) -> Ctx:
    """FastAPI dependency for getting the request context object (Ctx) for API routes.

    Use the CtxDep convenience alias below for declaring dependencies in routes, e.g. `ctx: utils.CtxDep`.

    Args:
        request: The FastAPI Request object.
        common: The CommonParams for the request.

    Returns:
        The Ctx object containing the request, common parameters, cache client, database helper, and CWB instance.
    """
    cache = request.app.state.memcached
    db = request.app.state.db
    cwb = request.app.state.cwb
    return Ctx(request=request, common=common, cache=cache, db=db, cwb=cwb)


# Convenience type alias for declaring Ctx dependencies in routes
CtxDep = Annotated[Ctx, Depends(get_ctx)]


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


def enforce_ctx_dependency(
    app: FastAPI,
) -> None:
    """Strictly enforce that every APIRoute endpoint has the required 'ctx' parameter.

    Every route is expected to have a parameter named 'ctx' or '_ctx' with the annotation 'utils.CtxDep', which injects
    the request context, containing common parameters and other commonly used objects.

    Raises:
        RuntimeError: If any route is missing the required 'ctx' parameter or has incorrect annotation.
    """
    param_name = "ctx"
    ctx_dependency = CtxDep
    ctx_dependency_name = "utils.CtxDep"  # For error messages
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


@dataclass
class AbortSignal:
    """Abort signal usable from both threads and async code.

    Routes can accept an `abort_signal: utils.AbortDep` parameter to get notified of client disconnects. Use
    `abort_signal.is_set()` to check if abort is requested, and await `abort_signal.wait()` to wait for it
    asynchronously.

    We need both threading.Event and asyncio.Event since some code runs in threads (e.g., CWB calls) while other code
    runs in the event loop.
    """

    _thread_evt: threading.Event
    _async_evt: asyncio.Event
    _loop: asyncio.AbstractEventLoop

    def set(self) -> None:
        """Set the abort signal."""
        self._thread_evt.set()
        with contextlib.suppress(Exception):
            self._loop.call_soon_threadsafe(self._async_evt.set)

    def is_set(self) -> bool:
        """Return True if the abort signal is set."""
        return self._thread_evt.is_set()

    async def wait(self) -> None:
        """Wait until the abort signal is set."""
        await self._async_evt.wait()


def abort_signal_dep() -> None:
    """Dummy dependency for abort_signal parameter.

    The real AbortSignal is created in the api_handler decorator and injected there. We use this dummy dependency to
    declare the parameter in route signatures without it being treated as client-provided input.
    """
    return


AbortDep = Annotated[AbortSignal | None, Depends(abort_signal_dep)]


def _format_error(exc: BaseException, *, debug: bool) -> dict[str, Any]:
    """Format an error response dictionary.

    Returns:
        A dictionary representing the error response.
    """
    err: dict[str, Any] = {"ERROR": {"type": type(exc).__name__, "value": str(exc)}}
    if debug:
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        err["ERROR"]["traceback"] = [line.rstrip("\n") for line in tb]
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

        ctx: utils.CtxDep

    An optional parameter may also be declared for routes that need abort signaling:

        abort_signal: utils.AbortSignal = None

    To check if abort is requested, use `abort_signal.is_set()`.

    Decorated routes can either:
      - yield dict fragments for incremental output
      - return dict
      - return Response (bypasses decorator processing)

    The decorator can be used with or without parentheses:
        @api_handler
        @api_handler(cache_headers=False)

    Args:
        _callable: The route function to decorate.
        cache_headers: Whether to set HTTP cache headers on the response.
        keepalive_seconds: Interval in seconds for sending keepalive whitespace.

    Returns:
        The decorated route function.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Awaitable[Response]]:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Response:
            ctx: Ctx = kwargs.get("ctx") or kwargs["_ctx"]  # Support both "ctx" and "_ctx"
            request = ctx.request
            common = ctx.common
            route = request.url.path
            method = request.method

            # Check for unexpected query parameters
            forbid_extra_query_params(request)

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
                    yield (json.dumps({"time": time.perf_counter() - start})[1:] + "\n").encode("utf-8")
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

                    result_obj["time"] = time.perf_counter() - start
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

            return resp

        return wrapper

    # If called as @api_handler without parentheses
    if _callable is not None:
        return decorator(_callable)

    # If called as @api_handler(...) with parentheses
    return decorator


def sync_generator_to_dict(generator: Generator[dict, None, None]) -> dict:
    """Convert a sync generator yielding dicts to a single dict.

    Args:
        generator: Generator yielding dicts.

    Returns:
        A single dict containing all key-value pairs from the yielded dicts.
    """
    result: dict = {}
    for d in generator:
        result.update(d)
    return result


async def async_generator_to_dict(generator: AsyncGenerator[dict, None]) -> dict:
    """Convert an async generator yielding dicts to a single dict.

    Args:
        generator: Generator yielding dicts.

    Returns:
        A single dict containing all key-value pairs from the yielded dicts.
    """
    result = {}
    async for d in generator:
        if isinstance(d, dict):
            result.update(d)
    return result


def get_corpus_timestamps() -> dict[str, float]:
    """Get modification time of corpus registry files.

    Returns:
        A dictionary mapping corpus names to their modification timestamps.
    """
    return {f.name.upper(): f.stat().st_mtime for f in Path(settings.CWB_REGISTRY).glob("*")}


def get_corpus_config_timestamps() -> tuple[dict[str, float], float, float]:
    """Get modification time of corpus config files.

    Returns:
        A tuple containing:
        - A dictionary mapping corpus names to their config file modification timestamps.
        - The latest modification timestamp among mode config files.
        - The latest modification timestamp among preset config files.
    """
    corpora = {
        f.name[:-5].upper(): f.stat().st_mtime for f in Path(settings.CORPUS_CONFIG_DIR, "corpora").glob("*.yaml")
    }
    modes = max((f.stat().st_mtime for f in Path(settings.CORPUS_CONFIG_DIR, "modes").glob("*.yaml")), default=0)
    presets = max(
        (f.stat().st_mtime for f in Path(settings.CORPUS_CONFIG_DIR, "attributes").glob("*/*.yaml")), default=0
    )
    return corpora, modes, presets


async def setup_cache(cache: Memcached) -> bool:
    """Setup disk cache and Memcached if needed.

    Args:
        cache: Memcached client.

    Returns:
        True if any action was needed (cache dir created or Memcached initialized), False otherwise.
    """
    action_needed = False

    # Create cache dir if needed
    if settings.CACHE_DIR and not Path(settings.CACHE_DIR).exists():
        Path(settings.CACHE_DIR).mkdir(parents=True)
        action_needed = True

    # Set up Memcached if needed
    if settings.MEMCACHED_SERVER and not await cache.get("multi:version"):
        memcached_data = {}
        corpora = get_corpus_timestamps()
        corpora_configs, config_modes, config_presets = get_corpus_config_timestamps()
        memcached_data["multi:version"] = 1
        memcached_data["multi:version_config"] = 1
        memcached_data["multi:corpora"] = set(corpora.keys())
        memcached_data["multi:config_corpora"] = set(corpora_configs.keys())
        memcached_data["multi:config_modes"] = config_modes
        memcached_data["multi:config_presets"] = config_presets
        for corpus in corpora:
            memcached_data[f"{corpus}:version"] = 1
            memcached_data[f"{corpus}:version_config"] = 1
            memcached_data[f"{corpus}:last_update"] = corpora[corpus]
            memcached_data[f"{corpus}:last_update_config"] = corpora_configs.get(corpus, 0)
        action_needed = True

        await cache.set_many(memcached_data)

    return action_needed


@overload
async def cache_prefix(
    cache: Memcached,
    corpus: str = "multi",
    config: bool = False,
) -> str: ...


@overload
async def cache_prefix(
    cache: Memcached,
    corpus: list[str],
    config: bool = False,
) -> dict[str, str]: ...


async def cache_prefix(
    cache: Memcached, corpus: str | list[str] = "multi", config: bool = False
) -> str | dict[str, str]:
    """Get cache version to use as prefix for cache keys.

    Args:
        cache: Memcached client.
        corpus: Corpus name or list of corpus names.
        config: Whether to get config version.

    Returns:
        Cache prefix string or dictionary of prefixes for multiple corpora.
    """
    if single := isinstance(corpus, str):
        corpus = [corpus]
    corpus_keys = {c: f"{c}:version{'_config' if config else ''}" for c in corpus}
    versions = await cache.get_many(corpus_keys.values())

    if single:
        return f"{corpus[0]}:{versions.get(corpus_keys[corpus[0]], 0)}"
    return {c: f"{c}:{versions.get(corpus_keys[c], 0)}" for c in corpus}


def cache_prefix_sync(
    cache: MemcachedSyncClient, corpus: str | list[str] = "multi", config: bool = False
) -> str | dict[str, str]:
    """Get cache version to use as prefix for cache keys (synchronous version).

    Args:
        cache: Memcached client.
        corpus: Corpus name or list of corpus names.
        config: Whether to get config version.

    Returns:
        Cache prefix string or dictionary of prefixes for multiple corpora.
    """
    if single := isinstance(corpus, str):
        corpus = [corpus]
    corpus_keys = {c: f"{c}:version{'_config' if config else ''}" for c in corpus}
    versions = cache.get_many(corpus_keys.values())

    if single:
        return f"{corpus[0]}:{versions.get(corpus_keys[corpus[0]], 0)}"
    return {c: f"{c}:{versions.get(corpus_keys[c], 0)}" for c in corpus}


# Pre-compiled patterns for wildcard/repetition parsing in query_optimize
_RE_WILDCARD_RANGE = re.compile(r"\{\s*(\d+)\s*,\s*(\d*)\s*\}$")
_RE_WILDCARD_EXACT = re.compile(r"\{\s*(\d*)\s*\}$")
_RE_REPETITION = re.compile(r"\{.*?\}$")
_WILDCARD_MAX = 9999  # Upper bound representing an unbounded wildcard range


def _parse_wildcard_repeat(token: str) -> tuple[int, int] | None:
    """Parse repetition counts from a wildcard token like `[]{2,5}`.

    Args:
        token: A wildcard token string (must start with `[]`).

    Returns:
        A (min, max) tuple, or None if the token has no parseable repetition.
    """
    if token == "[]":
        return 1, 1
    if m := _RE_WILDCARD_RANGE.search(token):
        return int(m.group(1)), int(m.group(2)) if m.group(2) else _WILDCARD_MAX
    if m := _RE_WILDCARD_EXACT.search(token):
        n = int(m.group(1))
        return n, n
    return None


class QueryOptimizeResult(Enum):
    """Result codes for query optimization."""

    SUCCESS = 0
    """Optimization successful; the query was transformed into an optimized MU query."""

    NOT_NEEDED = 1
    """Optimization not needed; the query is too simple to benefit from optimization (e.g., single word search)."""

    NOT_POSSIBLE = 2
    """Optimization not possible; the query contains constructs that prevent optimization (e.g., repetition of
    non-wildcards)."""


def optimize_query(
    cqp: str, cqp_params: dict, find_match: bool = True, expand: bool = True, free_search: bool = False
) -> tuple[QueryOptimizeResult, list[str]]:
    """Optimize simple queries with multiple words by converting them to MU queries.

    Optimization only works for queries with at least two tokens, or one token preceded by one or more wildcards. The
    query also must use `within`.

    Args:
        cqp: The CQP query string.
        cqp_params: Additional CQP parameters (within, cut, expand).
        find_match: Whether to mark all matching words in the result (not just the first).
        expand: Whether to expand the query.
        free_search: Whether the query is a free order search.

    Returns:
        A tuple containing:
        - A QueryOptimizeResult indicating the optimization outcome.
        - A list of strings representing the optimized query.

    Raises:
        CQPError: If the query cannot be optimized due to unsupported constructs.
    """
    tokens, rest = parse_cqp(cqp)
    within = cqp_params.get("within")
    fallback_query = make_query(make_cqp(cqp, **cqp_params))

    leading_wildcards = False

    if free_search:
        # Don't allow wildcards in free searches
        if any(token.startswith("[]") for token in tokens):
            raise CQPError("Wildcards not allowed in free order query.")
    else:
        # Strip leading and trailing wildcards since they only slow things down
        start = 0
        while start < len(tokens) and tokens[start].startswith("[]"):
            leading_wildcards = True
            start += 1
        end = len(tokens)
        while end > start and tokens[end - 1].startswith("[]"):
            end -= 1
        tokens = tokens[start:end]

    if not tokens or (len(tokens) == 1 and not leading_wildcards):
        # Query doesn't benefit from optimization
        return QueryOptimizeResult.NOT_NEEDED, fallback_query
    if rest or not within:
        # Couldn't optimize this query
        return QueryOptimizeResult.NOT_POSSIBLE, fallback_query

    # Build the MU command
    mu_parts: list[str] = ["MU"]
    wildcards: dict[int, tuple[int, int]] = {}

    for i, token in enumerate(tokens[:-1]):
        if token.startswith("[]"):
            repeat = _parse_wildcard_repeat(token)
            if repeat is not None:
                wildcards[i] = repeat
            continue
        if _RE_REPETITION.search(token):
            # Repetition for anything other than wildcards can't be optimized
            return QueryOptimizeResult.NOT_POSSIBLE, fallback_query
        mu_parts.append(f"(meet {token}")

    if _RE_REPETITION.search(tokens[-1]):
        return QueryOptimizeResult.NOT_POSSIBLE, fallback_query

    mu_parts.append(tokens[-1])

    # Build closing parts with distance constraints (reverse order)
    wc_min = wc_max = 1
    for i in range(len(tokens) - 2, -1, -1):
        if i in wildcards:
            wc_min += wildcards[i][0]
            wc_max += wildcards[i][1]
            continue
        if i + 1 in wildcards:
            mu_parts.append(f"{within})" if wc_max >= _WILDCARD_MAX else f"{wc_min} {wc_max})")
            wc_min = wc_max = 1
        elif free_search:
            mu_parts.append(f"{within})")
        else:
            mu_parts.append("1 1)")

    mu_cmd = " ".join(mu_parts)
    cmd: list[str] = []

    if find_match and not free_search:
        # MU searches only highlight the first keyword of each hit. To highlight all keywords we need to
        # do a new non-optimized search within the results, and to be able to do that we first need to expand the rows.
        # Most of the time we only need to expand to the right, except for when leading wildcards are used.
        direction = "expand to" if leading_wildcards else "expand right to"
        cmd.extend([f"{mu_cmd} {direction} {within};", "Last;", *fallback_query])
    elif expand or free_search:
        cmd.append(f"{mu_cmd} expand to {within};")
    else:
        cmd.append(f"{mu_cmd};")

    return QueryOptimizeResult.SUCCESS, cmd


def split_csv(values: str | Iterable[str] | None) -> list[str]:
    """Split comma-separated values into a list.

    Accepts a string (comma-separated) or an iterable of strings (repeated query params).
    Empty values are dropped. Order is preserved.

    Args:
        values: A string of comma-separated values, an iterable of strings, or None.

    Returns:
        A list of individual values.
    """
    if values is None:
        return []

    raw_values = [values] if isinstance(values, str) else list(values)
    return [item for raw in raw_values for item in raw.split(QUERY_DELIM) if item]


def parse_within(within: Sequence[str] | None, default_within: str | None = None) -> dict[str, str | None]:
    """Parse 'within' parameter into a dictionary mapping corpora to within values.

    Args:
        within: A sequence of 'CORPUS:WITHIN' pairs.
        default_within: The default within value to use for corpora not specified in the 'within' parameter.

    Returns:
        A dictionary mapping corpus names to their respective within values.

    Raises:
        ValueError: If the 'within' parameter is malformed.
    """
    within_dict = defaultdict(lambda: default_within)
    within = within or []

    for pair in within:
        if ":" not in pair:
            raise ValueError("Malformed value for key 'within'.")
        corpus, within_value = pair.split(":", 1)
        within_dict[corpus.upper()] = within_value
    return within_dict


def parse_cqp(cqp: str) -> tuple[list[str], bool]:
    """Try to parse a CQP query, returning identified tokens and a boolean indicating partial failure if True.

    This is used by the query optimizer, and by "free order" searches.

    Args:
        cqp: The CQP query string.

    Returns:
        A tuple containing:
            - A list of strings representing the identified tokens.
            - A boolean indicating whether the parsing was only partially successful.
    """
    cqp_len = len(cqp)
    sections: list[list[int]] = []
    last_start = 0
    in_bracket = False
    in_quote = False
    in_curly = False
    escaping = False
    quote_type = ""

    for i, c in enumerate(cqp):
        # Handle escape sequences (only relevant inside quotes)
        if escaping:
            escaping = False
            continue

        if in_quote:
            if c == "\\":
                escaping = True
            elif c == quote_type:
                if i + 1 < cqp_len and cqp[i + 1] == quote_type:
                    # Quote escaped by doubling
                    escaping = True
                else:
                    # End of a quote
                    in_quote = False
                    if not in_bracket:
                        sections.append([last_start, i])
            # Skip all bracket/curly checks when inside a quote
            continue

        # Outside quotes
        if c in "'\"":
            # Beginning of a quote
            in_quote = True
            quote_type = c
            if not in_bracket:
                last_start = i
        elif c == "[":
            if not in_bracket:
                # Beginning of a token
                last_start = i
                in_bracket = True
                if i + 1 < cqp_len and cqp[i + 1] == ":":
                    # Zero-width assertion encountered, which cannot be handled by MU query
                    return [], True
        elif c == "]":
            if in_bracket:
                # End of a token
                sections.append([last_start, i])
                in_bracket = False
        elif c == "{" and not in_bracket:
            in_curly = True
        elif c == "}" and not in_bracket and in_curly:
            in_curly = False
            sections[-1][1] = i

    # Build token list and detect non-token content ("rest") between sections
    sections.append([cqp_len, cqp_len])
    tokens: list[str] = []
    rest = False
    prev_end = 0

    for start, end in sections:
        if prev_end < start and cqp[prev_end + 1 : start].strip():
            rest = True
        prev_end = end
        token = cqp[start : end + 1]
        if token:
            tokens.append(token)

    return tokens, rest


def make_cqp(cqp: str, within: str | None = None, cut: str | None = None, expand: str | None = None) -> str:
    """Combine CQP query and extra options into a single CQP query string.

    Args:
        cqp: The CQP query string.
        within: The 'within' option.
        cut: The 'cut' option.
        expand: The 'expand' option.

    Returns:
        The combined CQP query string with options appended, and terminated with a semicolon.
    """
    parts = [cqp]
    if within:
        parts.append(f"within {within}")
    if cut:
        parts.append(f"cut {cut}")
    if expand:
        parts.append(f"expand {expand}")
    return " ".join(parts) + ";"


def make_query(cqp: str | list[str]) -> list[str]:
    """Create web-safe commands for a CQP query.

    This wraps the CQP query with commands to enable and disable query lock mode. This prevents execution of arbitrary
    commands, allowing only queries to be executed.

    Args:
        cqp: The CQP query string or list of CQP query strings. Each string must be terminated with a semicolon.

    Returns:
        A list of CQP commands with query lock enabled.
    """
    querylock = random.randrange(10**8, 10**9)
    if isinstance(cqp, str):
        cqp = [cqp]
    return [f"set QueryLock {querylock};", *cqp, f"unlock {querylock};"]


def translate_undef(s: str | None) -> str | None:
    """Translate '__UNDEF__' to None.

    '__UNDEF__' can be used in corpora to represent undefined values.

    Args:
        s: The input string.

    Returns:
        `None` if the input string is '__UNDEF__', otherwise the original string.
    """
    return None if s == UNDEF_VALUE else s


def get_hash(values: Iterable[Any]) -> str:
    """Get a hash for a list of values.

    Args:
        values: A list of values to hash.

    Returns:
        A SHA-256 hash of the concatenated values.
    """
    return hashlib.sha256(";".join(v if isinstance(v, str) else str(v) for v in values).encode()).hexdigest()


class CQPError(Exception):
    """Custom exception for CQP errors."""


class KorpAuthorizationError(Exception):
    """Custom exception for Korp authorization errors."""


class Namespace(SimpleNamespace):
    """Simple namespace class to hold attributes."""


def _make_auth_context(ctx: Ctx) -> AuthContext:
    """Create an AuthContext from a request context.

    Returns:
        An AuthContext with request and cache settings from the given context.
    """
    return AuthContext(request=ctx.request, cache_enabled=ctx.common.cache)


async def get_protected_corpora(ctx: Ctx) -> list[str]:
    """Return a list of corpora with restricted access."""
    authorizer = ctx.request.app.state.authorizer
    if authorizer:
        return await authorizer.get_protected_corpora(_make_auth_context(ctx))
    return []


async def check_authorization(corpora: Iterable[str], ctx: Ctx) -> None:
    """Take a list of corpora, and if any of them are protected, check authorization.

    Args:
        corpora: List of corpus names.
        ctx: Request context used to build authorizer context.

    Raises:
        KorpAuthorizationError: If the user is not authorized to access one or more of the specified corpora.
    """
    authorizer = ctx.request.app.state.authorizer
    if authorizer:
        # Split parallel corpora
        corpora = [cc for c in corpora for cc in c.split("|")]

        success, unauthorized, message = await authorizer.check_authorization(corpora, _make_auth_context(ctx))
        if not success:
            if not message:
                message = "You do not have access to the following corpora: {}".format(", ".join(unauthorized))
            raise KorpAuthorizationError(message)


def strptime(date: str) -> datetime.datetime:
    """Take a date in string format and return a datetime object.

    We need this since the built-in strptime isn't thread safe (and this is much faster).

    Args:
        date: Date string in the format "YYYYMMDDhhmmss".

    Returns:
        A datetime object representing the parsed date.
    """
    year = int(date[:4])
    month = int(date[4:6]) if len(date) > 4 else 1  # noqa: PLR2004
    day = int(date[6:8]) if len(date) > 6 else 1  # noqa: PLR2004
    hour = int(date[8:10]) if len(date) > 8 else 0  # noqa: PLR2004
    minute = int(date[10:12]) if len(date) > 10 else 0  # noqa: PLR2004
    second = int(date[12:14]) if len(date) > 12 else 0  # noqa: PLR2004
    return datetime.datetime(year, month, day, hour, minute, second)


def sql_escape(s: str) -> str:
    """Return SQL-escaped version of string s."""
    return _db_escape_string(s) if isinstance(s, str) else s


class Plugin(APIRouter):
    """Simple plugin class compatible with FastAPI's router API."""

    def __init__(self, name: str, import_name: str, **kwargs: Any) -> None:
        """Initialize plugin.

        Args:
            name: Plugin name.
            import_name: Plugin import name.
            **kwargs: Additional keyword arguments for APIRouter.
        """
        super().__init__(**kwargs)
        self.name = name
        self.import_name = import_name

    def config(self, key: str, default: Any = None) -> Any:
        """Get plugin configuration value.

        Args:
            key: Configuration key.
            default: Default value if key is not found.

        Returns:
            The configuration value or default.
        """
        return settings.PLUGINS_CONFIG.get(self.import_name, {}).get(key, default)


class Authorizer(ABC):
    """Class to subclass when implementing an authorizer plugin.

    The authorizer is responsible for determining which corpora have restricted access, and for checking whether a user
    is authorized to access a given list of corpora. The authorizer can use any information available in the request
    context, such as headers or cookies, to make these determinations. The authorizer can also use the CWB and cache to
    look up information about corpora or users if needed.

    When creating an authorizer plugin, you must define a module-level variable `AUTHORIZER_CLASS` that references your
    Authorizer subclass.
    """

    def __init__(self, cwb: CWB, cache: Memcached) -> None:
        """Initialize authorizer with app-scoped dependencies."""
        self.cwb = cwb
        self.cache = cache

    @abstractmethod
    async def get_protected_corpora(self, auth_ctx: AuthContext) -> list[str]:
        """Get list of corpora with restricted access, in uppercase."""

    @abstractmethod
    async def check_authorization(
        self, corpora: list[str], auth_ctx: AuthContext
    ) -> tuple[bool, list[str], str | None]:
        """Take a list of corpora and check that the user has permission to access them."""

"""FastAPI dependencies and context objects for the Korp API."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Query, Request

from korp.api.requests import CommonQueryControls

if TYPE_CHECKING:
    from korp.cwb import CWB
    from korp.db import MySQL
    from korp.memcached import Memcached


async def common_params(  # noqa: RUF029 - avoid unnecessarily running dependency in a worker thread
    request: Request,
    controls: Annotated[CommonQueryControls, Query()],
) -> CommonParams:
    """FastAPI dependency for common parameters, available in all routes through the Ctx object.

    Args:
        request: The FastAPI Request object.
        controls: Parsed response controls.

    Returns:
        A CommonParams object with the parsed parameters.
    """
    return build_common_params(request, controls)


def build_common_params(request: Request, controls: CommonQueryControls) -> CommonParams:
    """Build effective common parameters from parsed query controls.

    Returns:
        Common parameters with application-level caching applied.
    """
    effective_cache = request.app.state.cache_enabled and controls.cache

    return CommonParams(
        debug=controls.debug,
        indent=controls.indent,
        stream=controls.stream,
        cache=effective_cache,
    )


@dataclass
class CommonParams:
    """Common parameters for API routes.

    Attributes:
        debug: Whether to include debug information in responses.
        indent: Number of spaces to indent JSON output.
        stream: Whether to produce an NDJSON event stream.
        cache: Whether to use caching for the request.
    """

    debug: bool = False
    indent: int = 0
    stream: bool = False
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


async def get_ctx(  # noqa: RUF029
    request: Request,
    common: Annotated[CommonParams, Depends(common_params)],
) -> Ctx:
    """FastAPI dependency for getting the request context object (Ctx) for API routes.

    Use the CtxDep convenience alias below for declaring dependencies in routes, e.g. `ctx: CtxDep`.

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


async def get_query_ctx(request: Request) -> Ctx:  # noqa: RUF029
    """FastAPI dependency for getting the request context object (Ctx) for GET query routes.

    GET routes that use a query parameter model (Pydantic model annotated with `Query()`) instead of explicit parameters
    cannot use the `CtxDep`/`get_ctx` dependency. This is because `CtxDep` includes the `common_params` dependency,
    which is a query parameter model itself. FastAPI currently does not support multiple flattened query parameter
    models on the same endpoint. Therefore, GET routes with a query parameter model must use this dependency instead,
    which does not include the `common_params` dependency.

    Args:
        request: The FastAPI Request object.

    Returns:
        A context with placeholder `CommonParams`, replaced by `api_handler` before route execution.
    """
    return Ctx(
        request=request,
        common=CommonParams(),
        cache=request.app.state.memcached,
        db=request.app.state.db,
        cwb=request.app.state.cwb,
    )


# Convenience type alias for declaring Ctx dependencies in routes
CtxDep = Annotated[Ctx, Depends(get_ctx)]
QueryCtxDep = Annotated[Ctx, Depends(get_query_ctx)]


@dataclass
class AbortSignal:
    """Abort signal usable from both threads and async code.

    Routes can accept an `abort_signal: AbortDep` parameter to get notified of client disconnects. Use
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


async def abort_signal_dep() -> None:  # noqa: RUF029
    """Dummy dependency for abort_signal parameter.

    The real AbortSignal is created in the api_handler decorator and injected there. We use this dummy dependency to
    declare the parameter in route signatures without it being treated as client-provided input.
    """
    return


AbortDep = Annotated[AbortSignal | None, Depends(abort_signal_dep)]

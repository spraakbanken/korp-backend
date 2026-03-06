"""FastAPI dependencies and context objects for the Korp API."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Query, Request

if TYPE_CHECKING:
    from korp.cwb import CWB
    from korp.db import MySQL
    from korp.memcached import Memcached


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


# Convenience type alias for declaring Ctx dependencies in routes
CtxDep = Annotated[Ctx, Depends(get_ctx)]


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


def abort_signal_dep() -> None:
    """Dummy dependency for abort_signal parameter.

    The real AbortSignal is created in the api_handler decorator and injected there. We use this dummy dependency to
    declare the parameter in route signatures without it being treated as client-provided input.
    """
    return


AbortDep = Annotated[AbortSignal | None, Depends(abort_signal_dep)]

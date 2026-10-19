"""Miscellaneous routes."""

import asyncio
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter
from pydantic import Field

from korp.api import schemas
from korp.dependencies import CtxDep
from korp.handler import api_handler, docs_error_responses

router = APIRouter()


HEALTH_DESCRIPTION = """Return a minimal health check response.

This route is intended for monitoring systems. It does not perform deeper dependency checks against CQP, the database,
or Memcached; it only confirms that the FastAPI application can accept and answer a request.
"""


class HealthResponse(schemas.ResponseModel):
    """Response model for `/health` route."""

    status: Literal["ok"] = Field(..., description="Service status.", examples=["ok"])


@router.get(
    "/health",
    response_model=None,
    responses={
        200: {"model": HealthResponse},
        **docs_error_responses(),
    },
    summary="Health Check",
    description=HEALTH_DESCRIPTION,
    tags=["Administration"],
    operation_id="get_health",
)
async def health(_ctx: CtxDep) -> dict:
    """Health check endpoint for monitoring.

    This route intentionally does not use `api_handler`, as it is intended to be a minimal health check endpoint. It
    consequently does not support the NDJSON streaming response format.

    Returns:
        A dictionary with the health status.
    """
    return {"status": "ok"}


@router.get("/sleep", response_model=dict[str, int], include_in_schema=False)
@router.post("/sleep", response_model=dict[str, int], include_in_schema=False)
@api_handler
async def sleep(_ctx: CtxDep, t: int = 5) -> AsyncIterator[dict]:
    """Sleep for t seconds, yielding a value each second.

    This is mainly for testing purposes, particularly for demonstrating streamed responses.

    Args:
        t: Number of seconds to sleep.

    Yields:
        A dictionary with the current second count.
    """
    for x in range(t):
        await asyncio.sleep(1)
        yield {f"{x}": x}

"""Miscellaneous routes."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Query

from korp import cqp
from korp.dependencies import CtxDep
from korp.handler import api_handler

router = APIRouter()


@router.post("/health", response_model=dict, include_in_schema=False)
@router.get("/health", response_model=dict, tags=["Administration"])
async def health(_ctx: CtxDep) -> dict:
    """Health check endpoint for monitoring.

    Returns:
        A dictionary with the health status.
    """
    return {"status": "ok"}


@router.get("/sleep", response_model=dict[str, int], include_in_schema=False)
@router.post("/sleep", response_model=dict[str, int], include_in_schema=False)
@api_handler
async def sleep(_ctx: CtxDep, t: int = 5) -> AsyncIterator[dict]:
    """Sleep for t seconds, yielding a value each second.

    This is mainly for testing purposes, particularly for demonstrating incremental responses.

    Args:
        t: Number of seconds to sleep.

    Yields:
        A dictionary with the current second count.
    """
    for x in range(t):
        await asyncio.sleep(1)
        yield {f"{x}": x}


@router.get("/optimize", response_model=None)
@router.post("/optimize", response_model=None, include_in_schema=False)
@api_handler
async def optimize(
    _ctx: CtxDep,
    cqp_query: Annotated[str, Query(description="The CQP query to optimize.", alias="cqp")],
    within: str | None = None,
    cut: int | None = None,
    in_order: bool = True,
) -> AsyncGenerator[dict]:
    """Optimize a CQP query.

    Args:
        cqp_query: The CQP query to optimize.
        within: The structural unit to limit the search to.
        cut: The maximum number of hits.
        in_order: Whether the query terms should be in order.

    Yields:
        A dictionary with the optimized CQP query (or the original if optimization was not possible) and the
            optimization status.
    """
    cqp_params: dict[str, str | int] = {"within": within or "sentence"}
    if cut is not None:
        cqp_params["cut"] = cut

    free_search = not in_order

    optimization_status, optimized_cqp = cqp.optimize_query(
        cqp_query, cqp_params, find_match=False, expand=False, free_search=free_search
    )

    result = {
        "cqp": optimized_cqp if optimization_status == cqp.QueryOptimizeResult.SUCCESS else cqp_query,
        "status": optimization_status.name,
    }
    yield result

"""Miscellaneous routes."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator

from fastapi import APIRouter

from korp import utils

router = APIRouter()


@router.get("/sleep", response_model=dict[str, int], include_in_schema=False)
@router.post("/sleep", response_model=dict[str, int], include_in_schema=False)
@utils.api_handler
async def sleep(_ctx: utils.CtxDep, t: int = 5) -> AsyncIterator[dict]:
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
@utils.api_handler
async def optimize(
    _ctx: utils.CtxDep,
    cqp: str,
    within: str | None = None,
    cut: int | None = None,
    in_order: bool = True,
) -> AsyncGenerator[dict]:
    """Optimize a CQP query.

    Args:
        cqp: The CQP query to optimize.
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

    optimization_status, optimized_cqp = utils.optimize_query(
        cqp, cqp_params, find_match=False, expand=False, free_search=free_search
    )

    result = {
        "cqp": optimized_cqp if optimization_status == utils.QueryOptimizeResult.SUCCESS else cqp,
        "status": optimization_status.name,
    }
    yield result

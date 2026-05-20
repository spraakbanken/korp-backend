"""Miscellaneous routes."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Annotated, Literal, TypeAlias

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from korp import cqp
from korp.api import schemas
from korp.dependencies import CtxDep
from korp.handler import api_handler, docs_response

router = APIRouter()


HEALTH_DESCRIPTION = """Return a minimal health check response.

This route is intended for monitoring systems. It does not perform deeper dependency checks against CQP, the database,
or Memcached; it only confirms that the FastAPI application can accept and answer a request.
"""

OPTIMIZE_DESCRIPTION = """Rewrite a CQP query into a more efficient form when possible.

The optimizer targets simple multi-token searches that can be transformed into CQP's MU query form. The response
contains the optimized query when optimization succeeds, or the original query when optimization is not needed or not
possible.

Use `within` to tell the optimizer which structural unit the query should stay inside. Set `in_order=false` for
free-order searches, where the matched query tokens may occur in any order inside the structural unit.

It is not necessary to use this route before every search, as Korp automatically optimizes queries internally when
possible. This route is mainly intended for testing and for users who want to see the optimized CQP query that Korp
would generate for a given input query.
"""

OptimizeCQPParam: TypeAlias = Annotated[
    str,
    Query(
        description="CQP query to optimize.",
        alias="cqp",
        examples=['"och" [] [pos="NN"]'],
    ),
]

OptimizeWithinParam: TypeAlias = Annotated[
    str | None,
    Query(
        description=("Structural unit the optimized query should stay inside. Defaults to `sentence` when omitted."),
        examples=["sentence"],
    ),
]

OptimizeInOrderParam: TypeAlias = Annotated[
    bool,
    Query(
        description=("Whether token order should matter. Set to `false` to optimize the query as a free-order search.")
    ),
]


class HealthResponse(BaseModel):
    """Response model for `/health` route."""

    status: Literal["ok"] = Field(..., description="Service status.", examples=["ok"])


class OptimizeResponse(schemas.CommonResponse):
    """Response model for `/optimize` route."""

    cqp: str | list[str] = Field(
        ...,
        description=(
            "Optimized CQP query when optimization succeeds, otherwise the original query. Successful optimization "
            "may return a list of CQP statements."
        ),
        examples=[['MU(meet "och" [pos="NN"] 2 2);']],
    )
    status: Literal["SUCCESS", "NOT_NEEDED", "NOT_POSSIBLE"] = Field(
        ...,
        description="Optimization result.",
        examples=["SUCCESS"],
    )


@router.post("/health", response_model=None, include_in_schema=False)
@router.get(
    "/health",
    response_model=None,
    responses=docs_response(HealthResponse),
    summary="Health Check",
    description=HEALTH_DESCRIPTION,
    tags=["Administration"],
)
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


@router.get(
    "/optimize",
    response_model=None,
    responses=docs_response(OptimizeResponse),
    summary="Optimize CQP Query",
    description=OPTIMIZE_DESCRIPTION,
    tags=["Miscellaneous"],
)
@router.post("/optimize", response_model=None, include_in_schema=False)
@api_handler
async def optimize(
    _ctx: CtxDep,
    cqp_query: OptimizeCQPParam,
    within: OptimizeWithinParam = None,
    in_order: OptimizeInOrderParam = True,
) -> AsyncGenerator[dict]:
    """Optimize a CQP query.

    Args:
        cqp_query: The CQP query to optimize.
        within: The structural unit to limit the search to.
        in_order: Whether the query terms should be in order.

    Yields:
        A dictionary with the optimized CQP query (or the original if optimization was not possible) and the
            optimization status.
    """
    cqp_params: dict[str, str | int] = {"within": within or "sentence"}
    free_search = not in_order

    optimization_status, optimized_cqp = cqp.optimize_query(
        cqp_query, cqp_params, find_match=False, expand=False, free_search=free_search
    )

    result = {
        "cqp": optimized_cqp if optimization_status == cqp.QueryOptimizeResult.SUCCESS else cqp_query,
        "status": optimization_status.name,
    }
    yield result

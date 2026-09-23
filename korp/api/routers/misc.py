"""Miscellaneous routes."""

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated, Literal, TypeAlias

from fastapi import APIRouter, Query
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema

from korp import cqp
from korp.api import schemas
from korp.api.requests import QueryRequestModel, RequestModel
from korp.dependencies import CtxDep, QueryCtxDep
from korp.handler import api_handler, docs_error_responses, docs_response

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
    str | SkipJsonSchema[None],
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


class HealthResponse(schemas.ResponseModel):
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


class OptimizeRequest(RequestModel):
    """Request for CQP query optimization."""

    cqp_query: OptimizeCQPParam
    within: OptimizeWithinParam = None
    in_order: OptimizeInOrderParam = True


class OptimizeQuery(QueryRequestModel, OptimizeRequest):
    """GET query for CQP query optimization."""


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


@router.get(
    "/optimize",
    response_model=None,
    responses=docs_response(OptimizeResponse, cqp_error=True),
    summary="Optimize CQP Query",
    description=OPTIMIZE_DESCRIPTION,
    tags=["Miscellaneous"],
    operation_id="get_optimize",
)
@api_handler
async def optimize_get(
    _ctx: QueryCtxDep,
    query: Annotated[OptimizeQuery, Query()],
) -> dict:
    """Optimize a query.

    Returns:
        The optimization result.
    """
    return _optimize(query.to_request(OptimizeRequest))


@router.post(
    "/optimize",
    response_model=None,
    responses=docs_response(OptimizeResponse, cqp_error=True),
    summary="Optimize CQP Query",
    description=OPTIMIZE_DESCRIPTION,
    tags=["Miscellaneous"],
    operation_id="post_optimize",
)
@api_handler
async def optimize_post(_ctx: CtxDep, request: OptimizeRequest) -> dict:
    """Optimize a query.

    Returns:
        The optimization result.
    """
    return _optimize(request)


def _optimize(request: OptimizeRequest) -> dict:
    """Optimize a CQP query.

    Args:
        request: Validated optimization request.

    Returns:
        A dictionary with the optimized CQP query (or the original if optimization was not possible) and the
            optimization status.
    """
    cqp_params: dict[str, str | int] = {"within": request.within or "sentence"}
    free_search = not request.in_order

    optimization_status, optimized_cqp = cqp.optimize_query(
        request.cqp_query, cqp_params, find_match=False, expand=False, free_search=free_search
    )

    return {
        "cqp": optimized_cqp if optimization_status == cqp.QueryOptimizeResult.SUCCESS else request.cqp_query,
        "status": optimization_status.name,
    }

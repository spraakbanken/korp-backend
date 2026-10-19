"""Query translation and CQP optimization routes."""

from typing import Annotated, Any, Literal, TypeAlias

from fastapi import APIRouter, Query
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema

from korp import cqp
from korp.api import params, schemas
from korp.api.requests import QueryRequestModel, RequestModel
from korp.dependencies import CtxDep, QueryCtxDep
from korp.handler import api_handler, docs_response
from korp.query_translation import translate_query

router = APIRouter(prefix="/query", tags=["Query"])
translation_router = APIRouter(prefix="/query", tags=["Query"])

OPTIMIZE_DESCRIPTION = """Rewrite a CQP query into a more efficient form when possible.

The optimizer targets simple multi-token searches that can be transformed into CQP's MU query form. The response
contains the optimized query when optimization succeeds, or the original query when optimization is not needed or not
possible.

Use `within` to tell the optimizer which structural unit the query should stay inside. Set `free_order=true` for
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

OptimizeFreeOrderParam: TypeAlias = Annotated[
    bool,
    Query(description=("Set to `true` to optimize the query as a free-order search.")),
]


class OptimizeResponse(schemas.CommonResponse):
    """Response model for `/query/optimize` route."""

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
    free_order: OptimizeFreeOrderParam = False


class OptimizeQuery(QueryRequestModel, OptimizeRequest):
    """GET query for CQP query optimization."""


@router.get(
    "/optimize",
    response_model=None,
    responses=docs_response(OptimizeResponse, cqp_error=True),
    summary="Optimize CQP Query",
    description=OPTIMIZE_DESCRIPTION,
    operation_id="get_query_optimize",
)
@api_handler
async def optimize_get(_ctx: QueryCtxDep, query: Annotated[OptimizeQuery, Query()]) -> dict:
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
    operation_id="post_query_optimize",
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
    free_order = request.free_order

    optimization_status, optimized_cqp = cqp.optimize_query(
        request.cqp_query, cqp_params, find_match=False, expand=False, free_order=free_order
    )

    return {
        "cqp": optimized_cqp if optimization_status == cqp.QueryOptimizeResult.SUCCESS else request.cqp_query,
        "status": optimization_status.name,
    }


TRANSLATE_DESCRIPTION = "Translate one alternative-language query into a CQP expression."


class TranslationRequest(RequestModel):
    """Input to the query translation route."""

    query: params.NonEmptyString
    query_language: params.NonEmptyString


class TranslationQuery(QueryRequestModel, TranslationRequest):
    """GET query for the query translation route."""


class TranslationResponse(schemas.CommonResponse):
    """Response model for `/query/translate` route."""

    cqp: str


@translation_router.get(
    "/translate",
    response_model=None,
    responses=docs_response(TranslationResponse),
    summary="Translate Query",
    description=TRANSLATE_DESCRIPTION,
    operation_id="get_query_translate",
)
@api_handler
async def translate_get(ctx: QueryCtxDep, query: Annotated[TranslationQuery, Query()]) -> dict[str, Any]:
    """Translate a query using the registered provider.

    Returns:
        A dictionary with the translated CQP query.
    """
    return await _translate_query(ctx, query.query_language, query.query)


@translation_router.post(
    "/translate",
    response_model=None,
    responses=docs_response(TranslationResponse),
    summary="Translate Query",
    description=TRANSLATE_DESCRIPTION,
    operation_id="post_query_translate",
)
@api_handler
async def translate_post(ctx: CtxDep, request: TranslationRequest) -> dict[str, Any]:
    """Translate a query using the registered provider.

    Returns:
        A dictionary with the translated CQP query.
    """
    return await _translate_query(ctx, request.query_language, request.query)


async def _translate_query(ctx: CtxDep, query_language: str, query: str) -> dict[str, Any]:
    """Translate a query using the registered provider.

    Args:
        ctx: Request context.
        query_language: Language of the query.
        query: Query to translate.

    Returns:
        A dictionary with the translated CQP query.
    """
    result = await translate_query(ctx, query_language, query)
    return {"cqp": result}

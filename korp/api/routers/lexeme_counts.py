"""Router for lexeme counts."""

from collections.abc import AsyncIterator
from typing import Annotated, TypeAlias

from fastapi import APIRouter, Query
from pydantic import AfterValidator, BeforeValidator, Field
from pydantic.json_schema import SkipJsonSchema
from sqlalchemy import text

from korp import auth, utils
from korp.api import schemas
from korp.api.requests import QueryRequestModel, RequestModel
from korp.config import settings
from korp.dependencies import CtxDep, QueryCtxDep
from korp.handler import api_handler, docs_response

router = APIRouter(tags=["Statistics"])

LEXEME_COUNT_DESCRIPTION = """Return absolute frequencies for one or more lexemes.

The response contains a `lexeme_counts` object where each returned lexeme is a key and the value is the total frequency
in the selected corpora. If `corpora` is omitted, counts are summed over all corpora present in the lexeme counts table.

Only exact lexeme lookups are supported. Lexemes that are not found are omitted from the response rather than returned
with `0`.

### Example

Get the number of occurrences of two lexemes in one corpus:

`/lexeme-counts?lexemes=ge..vb.1,ta..vb.1&corpora=ROMI`
"""

CorporaParamOptional: TypeAlias = Annotated[
    list[str] | SkipJsonSchema[None],
    Query(
        description=("Corpora to count in. If omitted, counts are summed over all corpora in the lexeme counts table."),
        examples=[["ROMI"], ["ROMI", "SUC3"]],
    ),
    BeforeValidator(utils.split_csv),
    AfterValidator(lambda v: [x.upper() for x in v]),
]

LexemesParam: TypeAlias = Annotated[
    list[str],
    Query(
        description="Lexemes to look up.",
        examples=[["ge..vb.1", "ta..vb.1"]],
    ),
    BeforeValidator(utils.split_csv),
]


class LexemeCountResponse(schemas.CommonResponse):
    """Response model for `/lexeme-counts` route."""

    lexeme_counts: dict[str, int] = Field(
        ...,
        description="Frequencies keyed by lexeme. Lexemes that are not found are omitted.",
        examples=[{"ge..vb.1": 354, "ta..vb.1": 85}],
    )


class LexemeCountsRequest(RequestModel):
    """Request for lexeme counts."""

    json_array_fields = frozenset({"lexemes", "corpora"})

    lexemes: LexemesParam
    corpora: CorporaParamOptional = None


class LexemeCountsQuery(QueryRequestModel, LexemeCountsRequest):
    """GET query for lexeme counts."""

    csv_fields = LexemeCountsRequest.json_array_fields


async def _lexeme_counts_stream(ctx: CtxDep, lexemes: list[str], corpora: list[str]) -> AsyncIterator[dict]:
    """Query the database and yield lexeme counts.

    Yields:
        A dictionary with lexeme counts.
    """
    bind_params: dict[str, str] = {}
    lexeme_placeholders = ", ".join(f":lexeme_{i}" for i in range(len(lexemes)))
    for i, lexeme in enumerate(lexemes):
        bind_params[f"lexeme_{i}"] = lexeme

    corpora_sql = ""
    if corpora:
        corpus_placeholders = ", ".join(f":corpus_{i}" for i in range(len(corpora)))
        for i, c in enumerate(corpora):
            bind_params[f"corpus_{i}"] = c
        corpora_sql = f" AND corpus IN ({corpus_placeholders})"

    sql = text(f"""
        SELECT lexeme, SUM(freq) AS freq
        FROM {settings.DB_LEXEME_COUNTS_TABLE}
        WHERE lexeme IN ({lexeme_placeholders})
            {corpora_sql}
        GROUP BY lexeme
    """)

    async with ctx.db.async_connection() as conn:
        query_result = await conn.execute(sql, bind_params)
        yield {"lexeme_counts": {row["lexeme"]: int(row["freq"]) for row in query_result.mappings()}}


@router.get(
    "/lexeme-counts",
    response_model=None,
    responses=docs_response(LexemeCountResponse, http_errors={403: "Access to a requested corpus was denied."}),
    summary="Lexeme Statistics",
    description=LEXEME_COUNT_DESCRIPTION,
    operation_id="get_lexeme_counts",
)
@api_handler
async def lexeme_counts_get(
    ctx: QueryCtxDep,
    query: Annotated[LexemeCountsQuery, Query()],
) -> AsyncIterator[dict]:
    """Return lexeme counts for the given lexemes and corpora."""
    return await _lexeme_counts(ctx, query.to_request(LexemeCountsRequest))


@router.post(
    "/lexeme-counts",
    response_model=None,
    responses=docs_response(LexemeCountResponse, http_errors={403: "Access to a requested corpus was denied."}),
    summary="Lexeme Statistics",
    description=LEXEME_COUNT_DESCRIPTION,
    operation_id="post_lexeme_counts",
)
@api_handler
async def lexeme_counts_post(ctx: CtxDep, request: LexemeCountsRequest) -> AsyncIterator[dict]:
    """Return lexeme counts for the given lexemes and corpora."""
    return await _lexeme_counts(ctx, request)


async def _lexeme_counts(ctx: CtxDep, request: LexemeCountsRequest) -> AsyncIterator[dict]:
    """Return lexeme statistics per corpus.

    Args:
        ctx: Request context.
        request: Validated lexeme-count request.

    Returns:
        An async iterator yielding a dictionary with lexeme counts.
    """
    corpora = request.corpora or []
    await auth.check_authorization(corpora, ctx)
    return _lexeme_counts_stream(ctx, request.lexemes, corpora)

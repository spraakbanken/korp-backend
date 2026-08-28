"""Router for lexeme counts."""

from collections.abc import AsyncGenerator
from typing import Annotated, TypeAlias

from fastapi import APIRouter, Query
from pydantic import AfterValidator, BeforeValidator, Field
from pydantic.json_schema import SkipJsonSchema
from sqlalchemy import text

from korp import auth, utils
from korp.api import schemas
from korp.config import settings
from korp.dependencies import CtxDep
from korp.handler import api_handler, docs_response

router = APIRouter(tags=["Statistics"])

LEXEME_COUNT_DESCRIPTION = """Return absolute frequencies for one or more lexemes.

The response contains a `lexeme_counts` object where each returned lexeme is a key and the value is the total frequency
in the selected corpora. If `corpora` is omitted, counts are summed over all corpora present in the lexeme counts table.

Only exact lexeme lookups are supported. Lexemes that are not found are omitted from the response rather than returned
with `0`.

### Example

Get the number of occurrences of two lexemes in one corpus:

`/lexeme_counts?lexeme=ge..vb.1,ta..vb.1&corpora=ROMI`
"""

CorporaParamOptional: TypeAlias = Annotated[
    list[str] | SkipJsonSchema[None],
    Query(
        description=(
            "Comma-separated list of corpora to count in. If omitted, counts are summed over all corpora in the "
            "lexeme counts table."
        ),
        examples=[["ROMI"], ["ROMI,SUC3"]],
    ),
    BeforeValidator(utils.split_csv),
    AfterValidator(lambda v: [x.upper() for x in v]),
]

LexemeParam: TypeAlias = Annotated[
    list[str],
    Query(
        description="Comma-separated list of lexemes to look up.",
        examples=[["ge..vb.1,ta..vb.1"]],
    ),
    BeforeValidator(utils.split_csv),
]


class LexemeCountResponse(schemas.CommonResponse):
    """Response model for `/lexeme_counts` route."""

    lexeme_counts: dict[str, int] = Field(
        ...,
        description="Frequencies keyed by lexeme. Lexemes that are not found are omitted.",
        examples=[{"ge..vb.1": 354, "ta..vb.1": 85}],
    )


@router.get(
    "/lexeme_counts",
    response_model=None,
    responses=docs_response(LexemeCountResponse),
    summary="Lexeme Statistics",
    description=LEXEME_COUNT_DESCRIPTION,
)
@router.post("/lexeme_counts", response_model=None, include_in_schema=False)
@api_handler
async def lexeme_counts(
    ctx: CtxDep,
    lexeme: LexemeParam,
    corpora: CorporaParamOptional = None,
) -> AsyncGenerator[dict]:
    """Return lexeme statistics per corpus.

    Args:
        ctx: Request context.
        lexeme: Lexeme or multiple lexemes separated by the query delimiter.
        corpora: Comma-separated list of corpora.

    Yields:
        A dictionary with lexeme counts.
    """
    corpora = corpora or []
    await auth.check_authorization(corpora, ctx)

    bind_params: dict[str, str] = {}
    lexeme_placeholders = ", ".join(f":lexeme_{i}" for i in range(len(lexeme)))
    for i, l in enumerate(lexeme):
        bind_params[f"lexeme_{i}"] = l

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
        yield {
            "lexeme_counts": {
                row["lexeme"]: int(row["freq"]) for row in query_result.mappings()
            }
        }

"""Router for lemgram count."""

from collections.abc import AsyncGenerator
from typing import Annotated, TypeAlias

from fastapi import APIRouter, Query
from pydantic import AfterValidator, BeforeValidator, ConfigDict
from pydantic.json_schema import SkipJsonSchema
from sqlalchemy import text

from korp import auth, utils
from korp.api import schemas
from korp.dependencies import CtxDep
from korp.handler import api_handler, docs_response

router = APIRouter(tags=["Statistics"])

LEMGRAM_COUNT_DESCRIPTION = """Return absolute frequencies for one or more lemgrams.

The response is a flat object where each returned lemgram is a top-level key and the value is the total frequency in
the selected corpora. If `corpus` is omitted, counts are summed over all corpora present in the lemgram index.

Only exact lemgram lookups are supported. Lemgrams that are not found are omitted from the response rather than returned
with `0`.

### Example

Get the number of occurrences of two lemgrams in one corpus:

`/lemgram_count?lemgram=ge..vb.1,ta..vb.1&corpus=ROMI`
"""

CorpusParamOptional: TypeAlias = Annotated[
    list[str] | SkipJsonSchema[None],
    Query(
        description=(
            "Comma-separated list of corpora to count in. If omitted, counts are summed over all corpora in the "
            "lemgram index."
        ),
        examples=[["ROMI"], ["ROMI,SUC3"]],
    ),
    BeforeValidator(utils.split_csv),
    AfterValidator(lambda v: [x.upper() for x in v]),
]

LemgramParam: TypeAlias = Annotated[
    list[str],
    Query(
        description="Comma-separated list of lemgrams to look up.",
        examples=[["ge..vb.1,ta..vb.1"]],
    ),
    BeforeValidator(utils.split_csv),
]


class LemgramCountResponse(schemas.CommonResponse):
    """Response model for `/lemgram_count` route."""

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "description": "Lemgrams are returned as dynamic top-level keys with integer frequencies as values.",
            "additionalProperties": {"type": "integer"},
            "examples": [{"ge..vb.1": 354, "ta..vb.1": 85, "time": 0.0125}],
        },
    )


@router.get(
    "/lemgram_count",
    response_model=None,
    responses=docs_response(LemgramCountResponse),
    summary="Lemgram Statistics",
    description=LEMGRAM_COUNT_DESCRIPTION,
)
@router.post("/lemgram_count", response_model=None, include_in_schema=False)
@api_handler
async def lemgram_count(
    ctx: CtxDep,
    lemgram: LemgramParam,
    corpus: CorpusParamOptional = None,
) -> AsyncGenerator[dict]:
    """Return lemgram statistics per corpus.

    Args:
        ctx: Request context.
        lemgram: Lemgram or multiple lemgrams separated by the query delimiter.
        corpus: Corpus or multiple corpora separated by the query delimiter.

    Yields:
        A dictionary with lemgram counts.
    """
    corpora = corpus or []
    await auth.check_authorization(corpora, ctx)

    bind_params: dict[str, str] = {}
    lemgram_placeholders = ", ".join(f":lemgram_{i}" for i in range(len(lemgram)))
    for i, l in enumerate(lemgram):
        bind_params[f"lemgram_{i}"] = l

    corpora_sql = ""
    if corpora:
        corpus_placeholders = ", ".join(f":corpus_{i}" for i in range(len(corpora)))
        for i, c in enumerate(corpora):
            bind_params[f"corpus_{i}"] = c
        corpora_sql = f" AND corpus IN ({corpus_placeholders})"

    sql = text(f"""
        SELECT lemgram, SUM(freq) AS freq
        FROM lemgram_index
        WHERE lemgram IN ({lemgram_placeholders})
            {corpora_sql}
        GROUP BY lemgram
    """)

    async with ctx.db.async_connection() as conn:
        query_result = await conn.execute(sql, bind_params)
        result = {row["lemgram"]: int(row["freq"]) for row in query_result.mappings()}

    yield result

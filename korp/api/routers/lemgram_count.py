"""Router for lemgram count."""

from collections.abc import AsyncGenerator
from typing import Annotated, TypeAlias

from fastapi import APIRouter, Query
from pydantic import AfterValidator, BeforeValidator
from pydantic.json_schema import SkipJsonSchema
from sqlalchemy import text

from korp import utils

router = APIRouter()

CorpusParamOptional: TypeAlias = Annotated[
    list[str] | SkipJsonSchema[None],
    Query(description="Comma-separated list of corpora."),
    BeforeValidator(utils.split_csv),
    AfterValidator(lambda v: [x.upper() for x in v]),
]

LemgramParam: TypeAlias = Annotated[
    list[str],
    Query(description="Comma-separated list of lemgrams."),
    BeforeValidator(utils.split_csv),
]


@router.get("/lemgram_count", response_model=dict)
@router.post("/lemgram_count", response_model=dict, include_in_schema=False)
@utils.api_handler
async def lemgram_count(
    ctx: utils.CtxDep,
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
    utils.check_authorization(corpora, ctx)

    lemgrams_string = ", ".join(f"'{utils.sql_escape(l)}'" for l in set(lemgram))

    corpora_sql = (
        " AND corpus IN ({})".format(", ".join(f"'{utils.sql_escape(c)}'" for c in corpora)) if corpora else ""
    )

    sql = f"""
        SELECT
            lemgram, SUM(freq) AS freq
        FROM
            lemgram_index
        WHERE
            lemgram IN ({lemgrams_string})
            {corpora_sql}
        GROUP BY
            lemgram;
    """

    result = {}
    async with ctx.db.async_connection() as conn:
        query_result = await conn.execute(text(sql))
        for row in query_result.mappings():
            result[row["lemgram"]] = int(row["freq"])

    yield result

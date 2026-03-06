"""Router for lemgram count."""

from collections.abc import AsyncGenerator
from typing import Annotated, TypeAlias

from fastapi import APIRouter, Query
from pydantic import AfterValidator, BeforeValidator
from pydantic.json_schema import SkipJsonSchema
from sqlalchemy import text

from korp import auth, utils
from korp.dependencies import CtxDep
from korp.handler import api_handler

router = APIRouter(tags=["Statistics"])

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

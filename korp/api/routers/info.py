"""Routes for retrieving information about the Korp backend and available corpora."""

import importlib.metadata
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

from korp import auth, caching, cqp, handler, utils
from korp.api import params, schemas
from korp.dependencies import CtxDep
from korp.handler import api_handler
from korp.memcached import CacheError

router = APIRouter(tags=["Corpus Information"])

INFO_DESCRIPTION = """Get general information about the Korp backend.

The response contains the backend version, the CQP version reported by Corpus Workbench, all corpora available on the
server, and the subset of those corpora that require authorization.
"""

CORPUS_INFO_DESCRIPTION = """Fetch Corpus Workbench metadata for one or more corpora.

For each requested corpus, the response contains the positional, structural, and alignment CWB attributes reported by
CQP. These are the encoded fields used to expose corpus data, including annotations. The response also includes
key-value metadata from CQP's `info` command and the corpus `.info` file. Common metadata keys include
`Size`, `Sentences`, `Charset`, `FirstDate`, `LastDate`, and `Updated`, but installations may expose additional keys.

The `total_size` and `total_sentences` fields sum the corresponding values for the requested corpora.
"""


class InfoResponse(schemas.CommonResponse):
    """Response model for `/info` route."""

    version: str = Field(
        ..., description="Version of the Korp backend.", examples=[importlib.metadata.version("korp-backend")]
    )
    cqp_version: str = Field(..., description="CQP version reported by Corpus Workbench.", examples=["3.4.12"])
    corpora: list[str] = Field(
        ...,
        description="Corpus ids available on this backend.",
        examples=[["ROMI", "PAROLE"]],
    )
    protected_corpora: list[str] = Field(
        ...,
        description="Corpus ids from `corpora` that require authorization.",
        examples=[["CLASSIFIED", "MYDIARY"]],
    )


class CorpusAttributes(BaseModel):
    """CWB attribute names available in a corpus."""

    p: list[str] = Field(
        ...,
        description="Names of positional CWB attributes, i.e. token-level annotations.",
        examples=[["word", "lemma", "pos"]],
    )
    s: list[str] = Field(
        ...,
        description=(
            "Names of structural CWB attributes, usually sentence-, text-, or document-level annotations."
        ),
        examples=[["text", "text_id", "sentence", "sentence_id"]],
    )
    a: list[str] = Field(
        ...,
        description="Names of alignment CWB attributes for linked corpora.",
        examples=[["link_n"]],
    )


class CorpusWorkbenchInfo(BaseModel):
    """Corpus metadata returned by CQP."""

    model_config = ConfigDict(extra="allow")

    charset: str | SkipJsonSchema[None] = Field(
        None,
        alias="Charset",
        description="Character encoding of the corpus.",
        examples=["utf8"],
    )
    first_date: str | SkipJsonSchema[None] = Field(
        None,
        alias="FirstDate",
        description="Date and time of the oldest dated text in the corpus, if available.",
        examples=["1976-01-01 00:00:00"],
    )
    last_date: str | SkipJsonSchema[None] = Field(
        None,
        alias="LastDate",
        description="Date and time of the newest dated text in the corpus, if available.",
        examples=["1990-12-31 23:59:59"],
    )
    size: int = Field(
        ...,
        alias="Size",
        description="Number of tokens in the corpus, represented as a string by CQP.",
        examples=[2531038],
    )
    sentences: int | SkipJsonSchema[None] = Field(
        None,
        alias="Sentences",
        description="Number of sentences in the corpus, if available.",
        examples=[83643],
    )
    updated: str | SkipJsonSchema[None] = Field(
        None,
        alias="Updated",
        description="Date when the corpus was last updated, if available.",
        examples=["2018-05-13"],
    )


class CorpusInfoData(BaseModel):
    """Information for a single corpus."""

    attrs: CorpusAttributes = Field(..., description="CWB attribute names available in the corpus.")
    info: CorpusWorkbenchInfo = Field(..., description="Corpus metadata from CQP and the corpus `.info` file.")


class CorpusInfoResponse(schemas.CommonResponse):
    """Response model for `/corpora/info` route."""

    corpora: dict[str, CorpusInfoData] = Field(
        ...,
        description="Corpus information keyed by corpus id.",
    )
    total_size: int = Field(..., description="Total number of tokens in the requested corpora.", examples=[82762958])
    total_sentences: int = Field(
        ...,
        description="Total number of sentences in the requested corpora.",
        examples=[326556],
    )


@router.get("/", response_model=None, include_in_schema=False)
@router.get(
    "/info",
    name="General Information",
    response_model=None,
    responses=handler.docs_response(InfoResponse),
    summary="General Information",
    description=INFO_DESCRIPTION,
)
@router.post("/info", response_model=None, include_in_schema=False)
@api_handler
async def info(
    ctx: CtxDep,
) -> AsyncIterator[dict]:
    """Get general information about the Korp backend and available corpora.

    Yields:
        Info about the Korp backend and available corpora.
    """
    cache = ctx.cache
    cache_prefix = ""
    if ctx.common.cache:
        cache_prefix = await caching.cache_prefix(cache)
        result = await cache.get(f"{cache_prefix}:info")
        if result:
            if ctx.common.debug:
                result.setdefault("debug", {})
                result["debug"]["cache_read"] = True
            yield result
            return

    corpora = ctx.cwb.run_cqp("show corpora;")
    version = next(corpora)

    protected = await auth.get_protected_corpora(ctx)

    result = {
        "version": importlib.metadata.version("korp-backend"),
        "cqp_version": version,
        "corpora": list(corpora),
        "protected_corpora": protected,
    }

    if ctx.common.cache:
        try:
            added = await cache.add(f"{cache_prefix}:info", result)
        except CacheError:
            added = False
        if added and ctx.common.debug:
            result.setdefault("debug", {})
            result["debug"]["cache_saved"] = True

    yield result


@router.get(
    "/corpora/info",
    response_model=None,
    responses=handler.docs_response(CorpusInfoResponse),
    name="Corpus Information",
    summary="Corpus Information",
    description=CORPUS_INFO_DESCRIPTION,
)
@router.post("/corpora/info", response_model=None, include_in_schema=False)
@api_handler
async def corpus_info(
    ctx: CtxDep,
    corpora: params.CorporaParam,
) -> AsyncIterator[dict]:
    """Get information about a specific corpus or corpora.

    Args:
        ctx: The request context.
        corpora: Comma-separated list of corpora.

    Yields:
        Information about the specified corpus or corpora.
    """
    yield await get_corpus_info(ctx, corpora)


async def get_corpus_info(ctx: CtxDep, corpora: list[str], no_combined_cache: bool = False) -> dict:
    """Get information about a specific corpus or corpora.

    Args:
        ctx: The request context.
        corpora: List of corpora.
        no_combined_cache: If True, do not use combined caching for multiple corpora.

    Returns:
        Information about the specified corpus or corpora.
    """
    cache = ctx.cache
    save_cache: list[str] = []
    all_prefixes: dict[str, str] = {}
    combined_cache_key = ""

    if ctx.common.cache:
        all_prefixes = await caching.cache_prefix(cache, ["multi", *corpora])

        checksum_combined = utils.get_hash((sorted(corpora),))
        combined_cache_key = f"{all_prefixes['multi']}:info_{checksum_combined}"

        # Check if whole query is cached
        if cached_result := await cache.get(combined_cache_key):
            if ctx.common.debug:
                cached_result.setdefault("debug", {})
                cached_result["debug"]["cache_read"] = True
                cached_result["debug"]["checksum"] = checksum_combined
            return cached_result

    result: dict[str, Any] = {"corpora": {}}
    total_size = 0
    total_sentences = 0

    if ctx.common.cache:
        memcached_keys = {f"{all_prefixes[c]}:info": c for c in corpora}
        cached_corpora = await cache.get_many(memcached_keys.keys())

        for key, c in memcached_keys.items():
            if key in cached_corpora:
                result["corpora"][c] = cached_corpora[key]
            else:
                save_cache.append(c)

    uncached_corpora = [c for c in corpora if c not in result["corpora"]]
    memcached_data = {}

    if uncached_corpora:
        cmd: list[str] = []
        for c in uncached_corpora:
            cmd += [f"{c};"]
            cmd += ctx.cwb.show_attributes()
            cmd += ["info; .EOL.;"]

        cmd += ["exit;"]

        # Call the CQP binary
        lines = ctx.cwb.run_cqp(cmd)

        # Skip CQP version
        next(lines)

        for c in uncached_corpora:
            # Read attributes
            attrs = ctx.cwb.read_attributes(lines)

            # Corpus information
            info = {}

            for line in lines:
                if line == cqp.END_OF_LINE:
                    break
                if ":" in line and not line.endswith(":"):
                    infokey, infoval = (x.strip() for x in line.split(":", 1))
                    if infokey in {"Size", "Sentences"} and isinstance(infoval, str) and infoval.isdigit():
                        infoval = int(infoval)
                    info[infokey] = infoval

            result["corpora"][c] = {"attrs": attrs, "info": info}
            if c in save_cache:
                memcached_data[f"{all_prefixes[c]}:info"] = result["corpora"][c]

    for c in corpora:
        info = result["corpora"][c]["info"]
        total_size += info["Size"]
        sentences = info.get("Sentences", 0)
        if isinstance(sentences, int):
            total_sentences += sentences

    if memcached_data:
        await cache.set_many(memcached_data)

    result["total_size"] = total_size
    result["total_sentences"] = total_sentences

    if ctx.common.cache and not no_combined_cache:
        # Cache whole query
        try:
            saved = await cache.add(combined_cache_key, result)
        except CacheError:
            pass
        else:
            if saved and ctx.common.debug:
                result.setdefault("debug", {})
                result["debug"]["cache_saved"] = True
    return result

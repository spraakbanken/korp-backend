"""Routes for retrieving information about the Korp backend and available corpora."""

import importlib.metadata
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from pydantic import Field

from korp import auth, caching, cqp, handler, utils
from korp.api import params, schemas
from korp.dependencies import CtxDep
from korp.handler import api_handler
from korp.memcached import CacheError

router = APIRouter(tags=["Corpus Information"])


class InfoResponse(schemas.CommonResponse):
    """Response model for `/info` route."""

    version: str = Field(
        ..., description="Version of the Korp backend.", examples=[importlib.metadata.version("korp-backend")]
    )
    cqp_version: str = Field(..., description="Version of the CQP binary.", examples=["3.4.12"])
    corpora: list[str] = Field(..., description="List of available corpora.", examples=[["CORPUS1", "CORPUS2"]])
    protected_corpora: list[str] = Field(..., description="List of protected corpora.", examples=[["CORPUS1"]])


@router.get("/", response_model=None, include_in_schema=False)
@router.get(
    "/info",
    name="General Information",
    response_model=None,
    responses=handler.docs_response(InfoResponse),
    description="Get information about the Korp backend and available corpora.",
)
@router.post("/info", response_model=None, include_in_schema=False)
@api_handler
async def info(
    ctx: CtxDep,
) -> AsyncIterator[dict]:
    """Get version information about list of available corpora.

    Yields:
        Info about the Korp backend and available corpora.
    """
    cache = ctx.cache
    if ctx.common.cache:
        cache_prefix = await caching.cache_prefix(cache)
        result = await cache.get(f"{cache_prefix}:info")
        if result:
            if ctx.common.debug:
                result.setdefault("DEBUG", {})
                result["DEBUG"]["cache_read"] = True
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
            result.setdefault("DEBUG", {})
            result["DEBUG"]["cache_saved"] = True

    yield result


@router.get("/corpus_info", response_model=dict, name="Corpus Information")
@router.post("/corpus_info", response_model=dict, include_in_schema=False)
@api_handler
async def corpus_info(
    ctx: CtxDep,
    corpus: params.CorpusParam,
) -> AsyncIterator[dict]:
    """Get information about a specific corpus or corpora.

    Args:
        ctx: The request context.
        corpus: Comma-separated list of corpora.

    Yields:
        Information about the specified corpus or corpora.
    """
    yield await get_corpus_info(ctx, corpus)


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
    combined_cache_key = ""

    if ctx.common.cache:
        all_prefixes = await caching.cache_prefix(cache, ["multi", *corpora])

        checksum_combined = utils.get_hash((sorted(corpora),))
        combined_cache_key = f"{all_prefixes['multi']}:info_{checksum_combined}"

        # Check if whole query is cached
        if cached_result := await cache.get(combined_cache_key):
            if ctx.common.debug:
                cached_result.setdefault("DEBUG", {})
                cached_result["DEBUG"]["cache_read"] = True
                cached_result["DEBUG"]["checksum"] = checksum_combined
            return cached_result

    result: dict[str, Any] = {"corpora": {}}
    total_size = 0
    total_sentences = 0

    cmd = []

    if ctx.common.cache:
        memcached_keys = {f"{all_prefixes[c]}:info": c for c in corpora}
        cached_corpora = await cache.get_many(memcached_keys.keys())

        for key, c in memcached_keys.items():
            if key in cached_corpora:
                result["corpora"][c] = cached_corpora[key]
            else:
                save_cache.append(c)

    for c in corpora:
        if c not in result["corpora"]:
            cmd += [f"{c};"]
            cmd += ctx.cwb.show_attributes()
            cmd += ["info; .EOL.;"]

    if cmd:
        cmd += ["exit;"]

        # Call the CQP binary
        lines = ctx.cwb.run_cqp(cmd)

        # Skip CQP version
        next(lines)

    memcached_data = {}

    for c in corpora:
        if c in result["corpora"]:
            total_size += int(result["corpora"][c]["info"]["Size"])
            sentences = result["corpora"][c]["info"].get("Sentences", "")
            if sentences.isdigit():
                total_sentences += int(sentences)
            continue

        # Read attributes
        attrs = ctx.cwb.read_attributes(lines)

        # Corpus information
        info = {}

        for line in lines:
            if line == cqp.END_OF_LINE:
                break
            if ":" in line and not line.endswith(":"):
                infokey, infoval = (x.strip() for x in line.split(":", 1))
                info[infokey] = infoval
                if infokey == "Size":
                    total_size += int(infoval)
                elif infokey == "Sentences" and infoval.isdigit():
                    total_sentences += int(infoval)

        result["corpora"][c] = {"attrs": attrs, "info": info}
        if c in save_cache:
            memcached_data[f"{all_prefixes[c]}:info"] = result["corpora"][c]

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
                result.setdefault("DEBUG", {})
                result["DEBUG"]["cache_saved"] = True
    return result

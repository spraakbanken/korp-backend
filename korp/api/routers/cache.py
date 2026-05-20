"""Route handler for cache management."""

import time
from pathlib import Path

from fastapi import APIRouter
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema

from korp import caching
from korp.api import schemas
from korp.config import settings
from korp.dependencies import CtxDep
from korp.handler import api_handler, docs_response

router = APIRouter(tags=["Administration"])

CACHE_DESCRIPTION = """Refresh Korp's cache metadata and remove stale cache files.

This administration route compares the current corpus registry and corpus configuration files with the versions stored
in Memcached. When a corpus or configuration has changed, the corresponding cache version is incremented so later API
requests stop using stale cached data. The route also removes expired query-data files from the cache directory.

If caching is disabled, the response contains only the common response fields. During first-time cache setup,
`initial_setup` is returned and no invalidation counters are included.
"""


class CacheResponse(schemas.CommonResponse):
    """Response model for `/cache` route."""

    initial_setup: bool | SkipJsonSchema[None] = Field(
        None,
        description="Whether cache metadata was initialized for the first time.",
        examples=[True],
    )
    multi_invalidated: bool | SkipJsonSchema[None] = Field(
        None,
        description="Whether combined query caches were invalidated because corpus data changed.",
        examples=[False],
    )
    multi_config_invalidated: bool | SkipJsonSchema[None] = Field(
        None,
        description="Whether combined configuration caches were invalidated because corpus configuration changed.",
        examples=[True],
    )
    corpora_invalidated: int | SkipJsonSchema[None] = Field(
        None,
        description="Number of corpora whose data cache version was incremented.",
        examples=[1],
    )
    configs_invalidated: int | SkipJsonSchema[None] = Field(
        None,
        description="Number of corpus configurations whose cache version was incremented.",
        examples=[2],
    )
    files_removed: int | SkipJsonSchema[None] = Field(
        None,
        description="Number of stale cache files removed from the cache directory.",
        examples=[14],
    )


@router.get(
    "/cache",
    response_model=None,
    responses=docs_response(CacheResponse),
    summary="Refresh Cache",
    description=CACHE_DESCRIPTION,
)
@router.post("/cache", response_model=None, include_in_schema=False)
@api_handler
async def cache_handler(ctx: CtxDep) -> dict:
    """Check for updated corpora and invalidate caches where needed, and remove old cache files.

    Returns:
        A dictionary with cache invalidation results.
    """
    if not ctx.request.app.state.cache_enabled:
        return {}

    cache = ctx.cache
    assert settings.CACHE_DIR

    # Set up caching if needed
    if await caching.setup_cache(cache):
        return {"initial_setup": True}

    result = {
        "multi_invalidated": False,
        "multi_config_invalidated": False,
        "corpora_invalidated": 0,
        "configs_invalidated": 0,
        "files_removed": 0,
    }
    now = time.time()

    # Get modification times of corpus registry and config files
    corpora = caching.get_corpus_timestamps()
    corpora_configs, config_modes, config_presets = caching.get_corpus_config_timestamps()

    # Fetch all needed cache keys at once
    per_corpus_keys: list[str] = []
    for corpus in corpora:
        per_corpus_keys.extend(
            [
                f"{corpus}:last_update",
                f"{corpus}:last_update_config",
                f"{corpus}:version",
                f"{corpus}:version_config",
            ]
        )
    multi_keys = [
        "multi:version",
        "multi:corpora",
        "multi:version_config",
        "multi:config_corpora",
        "multi:config_modes",
        "multi:config_presets",
    ]
    cached = await cache.get_many(per_corpus_keys + multi_keys)

    memcached_data: dict[str, object] = {}

    # Invalidate cache for updated corpora
    for corpus, corpus_mtime in corpora.items():
        if cached.get(f"{corpus}:last_update", 0) < corpus_mtime:
            memcached_data[f"{corpus}:version"] = int(cached.get(f"{corpus}:version", 0)) + 1
            memcached_data[f"{corpus}:last_update"] = corpus_mtime
            result["corpora_invalidated"] += 1

            # Remove outdated query data
            for cachefile in Path(settings.CACHE_DIR).glob(f"{corpus}:*"):
                try:
                    if cachefile.stat().st_mtime < corpus_mtime:
                        cachefile.unlink()
                        result["files_removed"] += 1
                except FileNotFoundError:
                    pass

        config_mtime = corpora_configs.get(corpus, 0)
        if cached.get(f"{corpus}:last_update_config", 0) < config_mtime:
            memcached_data[f"{corpus}:version_config"] = int(cached.get(f"{corpus}:version_config", 0)) + 1
            memcached_data[f"{corpus}:last_update_config"] = config_mtime
            result["configs_invalidated"] += 1

    # If any corpus has been updated, added or removed, increase version to invalidate all combined caches
    if result["corpora_invalidated"] or cached.get("multi:corpora", set()) != set(corpora):
        memcached_data["multi:version"] = int(cached.get("multi:version", 0)) + 1
        memcached_data["multi:corpora"] = set(corpora)
        result["multi_invalidated"] = True

    # Have any config modes or presets been updated?
    configs_updated = config_modes > int(cached.get("multi:config_modes", 0)) or config_presets > int(
        cached.get("multi:config_presets", 0)
    )

    # If modes or presets have been updated, or any corpus config has been updated, added or removed, increase
    # version to invalidate all combined caches
    if (
        configs_updated
        or result["configs_invalidated"]
        or cached.get("multi:config_corpora", set()) != set(corpora_configs)
    ):
        memcached_data["multi:version_config"] = int(cached.get("multi:version_config", 0)) + 1
        memcached_data["multi:config_corpora"] = set(corpora_configs)
        memcached_data["multi:config_modes"] = config_modes
        memcached_data["multi:config_presets"] = config_presets
        result["multi_config_invalidated"] = True

    await cache.set_many(memcached_data)

    # Remove old query data
    for cachefile in Path(settings.CACHE_DIR).glob("*:query_data_*"):
        try:
            if cachefile.stat().st_mtime < (now - settings.CACHE_LIFESPAN * 60):
                cachefile.unlink()
                result["files_removed"] += 1
        except FileNotFoundError:
            pass

    return result

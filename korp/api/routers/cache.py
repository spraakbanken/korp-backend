"""Route handler for cache management."""

import time
from pathlib import Path

from fastapi import APIRouter

from korp import utils
from korp.config import settings

router = APIRouter()


@router.get("/cache", response_model=dict)
@router.post("/cache", response_model=dict, include_in_schema=False)
@utils.api_handler
async def cache_handler(ctx: utils.CtxDep) -> dict:
    """Check for updated corpora and invalidate caches where needed, and remove old cache files.

    Returns:
        A dictionary with cache invalidation results.
    """
    if not ctx.request.app.state.cache_enabled:
        return {}

    cache = ctx.cache

    # Set up caching if needed
    if await utils.setup_cache(cache):
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
    corpora = utils.get_corpus_timestamps()
    corpora_configs, config_modes, config_presets = utils.get_corpus_config_timestamps()

    # Fetch all needed cache keys at once
    per_corpus_keys: list[str] = []
    for corpus in corpora:
        per_corpus_keys.extend([
            f"{corpus}:last_update",
            f"{corpus}:last_update_config",
            f"{corpus}:version",
            f"{corpus}:version_config",
        ])
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

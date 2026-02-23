"""Route handler for cache management."""

import time
from pathlib import Path
from typing import cast

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

    result = {}
    cache = ctx.cache

    # Set up caching if needed
    initial_setup = await utils.setup_cache(cache)

    if initial_setup:
        result["initial_setup"] = True
    else:
        result = {
            "multi_invalidated": False,
            "multi_config_invalidated": False,
            "corpora_invalidated": 0,
            "configs_invalidated": 0,
            "files_removed": 0,
        }
        now = time.time()

        # Get modification time of corpus registry files
        corpora = utils.get_corpus_timestamps()
        # Get modification time of corpus config files
        corpora_configs, config_modes, config_presets = utils.get_corpus_config_timestamps()

        memcached_data = {}

        last_update_keys = {corpus: f"{corpus}:last_update" for corpus in corpora}
        last_update = await cache.get_many(last_update_keys.values())

        last_update_config_keys = {corpus: f"{corpus}:last_update_config" for corpus in corpora}
        last_update_config = await cache.get_many(last_update_config_keys.values())

        # Invalidate cache for updated corpora
        for corpus in corpora:
            if last_update.get(last_update_keys[corpus], 0) < corpora[corpus]:
                memcached_data[f"{corpus}:version"] = cast(int, await cache.get(f"{corpus}:version", 0)) + 1
                memcached_data[f"{corpus}:last_update"] = corpora[corpus]
                result["corpora_invalidated"] += 1

                # Remove outdated query data
                for cachefile in Path(settings.CACHE_DIR).glob(f"{corpus}:*"):
                    try:
                        if cachefile.stat().st_mtime < corpora[corpus]:
                            cachefile.unlink()
                            result["files_removed"] += 1
                    except FileNotFoundError:
                        pass

            if last_update_config.get(last_update_config_keys[corpus], 0) < corpora_configs.get(corpus, 0):
                memcached_data[f"{corpus}:version_config"] = (
                    cast(int, await cache.get(f"{corpus}:version_config", 0)) + 1
                )
                memcached_data[f"{corpus}:last_update_config"] = corpora_configs[corpus]
                result["configs_invalidated"] += 1

        # If any corpus has been updated, added or removed, increase version to invalidate all combined caches
        if result["corpora_invalidated"] or (await cache.get("multi:corpora", set())) != set(corpora.keys()):
            memcached_data["multi:version"] = cast(int, await cache.get("multi:version", 0)) + 1
            memcached_data["multi:corpora"] = set(corpora.keys())
            result["multi_invalidated"] = True

        # Have any config modes or presets been updated?
        configs_updated = config_modes > cast(int, await cache.get("multi:config_modes", 0)) or config_presets > cast(
            int, await cache.get("multi:config_presets", 0)
        )

        # If modes or presets have been updated, or any corpus config has been updated, added or removed, increase
        # version to invalidate all combined caches
        if (
            configs_updated
            or result["configs_invalidated"]
            or (await cache.get("multi:config_corpora", set())) != set(corpora_configs.keys())
        ):
            memcached_data["multi:version_config"] = cast(int, await cache.get("multi:version_config", 0)) + 1
            memcached_data["multi:config_corpora"] = set(corpora_configs.keys())
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

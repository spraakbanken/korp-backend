"""Cache management for Korp API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, overload

from korp.config import settings

if TYPE_CHECKING:
    from korp.memcached import Memcached, MemcachedSyncClient


def get_corpus_timestamps() -> dict[str, float]:
    """Get modification time of corpus registry files.

    Returns:
        A dictionary mapping corpus names to their modification timestamps.
    """
    assert settings.CWB_REGISTRY is not None  # Should be guaranteed by settings validation
    return {f.name.upper(): f.stat().st_mtime for f in Path(settings.CWB_REGISTRY).glob("*")}


def get_corpus_config_timestamps() -> tuple[dict[str, float], float, float]:
    """Get modification time of corpus config files.

    Returns:
        A tuple containing:
        - A dictionary mapping corpus names to their config file modification timestamps.
        - The latest modification timestamp among mode config files.
        - The latest modification timestamp among preset config files.
    """
    if not settings.CORPUS_CONFIG_DIR:
        return {}, 0, 0
    corpora = {
        f.name[:-5].upper(): f.stat().st_mtime for f in Path(settings.CORPUS_CONFIG_DIR, "corpora").glob("*.yaml")
    }
    modes = max((f.stat().st_mtime for f in Path(settings.CORPUS_CONFIG_DIR, "modes").glob("*.yaml")), default=0)
    presets = max(
        (f.stat().st_mtime for f in Path(settings.CORPUS_CONFIG_DIR, "attributes").glob("*/*.yaml")), default=0
    )
    return corpora, modes, presets


async def setup_cache(cache: Memcached) -> bool:
    """Setup disk cache and Memcached if needed.

    Args:
        cache: Memcached client.

    Returns:
        True if any action was needed (cache dir created or Memcached initialized), False otherwise.
    """
    action_needed = False

    # Create cache dir if needed
    if settings.CACHE_DIR and not Path(settings.CACHE_DIR).exists():
        Path(settings.CACHE_DIR).mkdir(parents=True)
        action_needed = True

    # Set up Memcached if needed
    if settings.MEMCACHED_SERVER and not await cache.get("multi:version"):
        memcached_data = {}
        corpora = get_corpus_timestamps()
        corpora_configs, config_modes, config_presets = get_corpus_config_timestamps()
        memcached_data["multi:version"] = 1
        memcached_data["multi:version_config"] = 1
        memcached_data["multi:corpora"] = set(corpora.keys())
        memcached_data["multi:config_corpora"] = set(corpora_configs.keys())
        memcached_data["multi:config_modes"] = config_modes
        memcached_data["multi:config_presets"] = config_presets
        for corpus in corpora:
            memcached_data[f"{corpus}:version"] = 1
            memcached_data[f"{corpus}:version_config"] = 1
            memcached_data[f"{corpus}:last_update"] = corpora[corpus]
            memcached_data[f"{corpus}:last_update_config"] = corpora_configs.get(corpus, 0)
        action_needed = True

        await cache.set_many(memcached_data)

    return action_needed


@overload
async def cache_prefix(
    cache: Memcached,
    corpus: str = "multi",
    config: bool = False,
) -> str: ...


@overload
async def cache_prefix(
    cache: Memcached,
    corpus: list[str],
    config: bool = False,
) -> dict[str, str]: ...


async def cache_prefix(
    cache: Memcached, corpus: str | list[str] = "multi", config: bool = False
) -> str | dict[str, str]:
    """Get cache version to use as prefix for cache keys.

    Args:
        cache: Memcached client.
        corpus: Corpus name or list of corpus names.
        config: Whether to get config version.

    Returns:
        Cache prefix string or dictionary of prefixes for multiple corpora.
    """
    if single := isinstance(corpus, str):
        corpus = [corpus]
    corpus_keys = {c: f"{c}:version{'_config' if config else ''}" for c in corpus}
    versions = await cache.get_many(corpus_keys.values())

    if single:
        return f"{corpus[0]}:{versions.get(corpus_keys[corpus[0]], 0)}"
    return {c: f"{c}:{versions.get(corpus_keys[c], 0)}" for c in corpus}


def cache_prefix_sync(
    cache: MemcachedSyncClient, corpus: str | list[str] = "multi", config: bool = False
) -> str | dict[str, str]:
    """Get cache version to use as prefix for cache keys (synchronous version).

    Args:
        cache: Memcached client.
        corpus: Corpus name or list of corpus names.
        config: Whether to get config version.

    Returns:
        Cache prefix string or dictionary of prefixes for multiple corpora.
    """
    if single := isinstance(corpus, str):
        corpus = [corpus]
    corpus_keys = {c: f"{c}:version{'_config' if config else ''}" for c in corpus}
    versions = cache.get_many(corpus_keys.values())

    if single:
        return f"{corpus[0]}:{versions.get(corpus_keys[corpus[0]], 0)}"
    return {c: f"{c}:{versions.get(corpus_keys[c], 0)}" for c in corpus}

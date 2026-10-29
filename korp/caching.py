"""Cache management for Korp API."""

from __future__ import annotations

import shlex
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, overload

from korp import utils
from korp.config import settings
from korp.corpus_config import get_corpus_config_files

if TYPE_CHECKING:
    from korp.memcached import Memcached, MemcachedSyncClient


def get_corpus_timestamps() -> dict[str, float]:
    """Get the latest modification time of each corpus registry and its declared info file.

    Returns:
        A dictionary mapping corpus names to their modification timestamps.
    """
    assert settings.CWB_REGISTRY is not None  # Should be guaranteed by settings validation
    corpora: dict[str, float] = {}
    for registry_file in Path(settings.CWB_REGISTRY).glob("*"):
        timestamp = registry_file.stat().st_mtime
        with registry_file.open(encoding="utf-8") as registry:
            for line in registry:
                fields = line.split(maxsplit=1)
                if not fields or fields[0] != "INFO":
                    continue

                # CWB paths may be double-quoted and contain escaped quotes or backslashes.
                _, info_path = shlex.split(line, comments=True)
                try:
                    info_timestamp = Path(info_path).stat().st_mtime
                except FileNotFoundError:
                    pass
                else:
                    timestamp = max(timestamp, info_timestamp)
                break

        corpora[utils.normalize_corpus_id(registry_file.name)] = timestamp
    return corpora


def get_corpus_config_timestamps() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Get modification time of corpus config files.

    Returns:
        A tuple containing:
        - A dictionary mapping corpus names to their config file modification timestamps.
        - Mode config paths mapped to modification timestamps.
        - Preset config paths mapped to modification timestamps.
    """
    if not settings.CORPUS_CONFIG_DIR:
        return {}, {}, {}
    corpora = {name: path.stat().st_mtime for name, path in get_corpus_config_files().items()}
    modes = {str(f): f.stat().st_mtime for f in Path(settings.CORPUS_CONFIG_DIR, "modes").glob("*.yaml")}
    presets = {str(f): f.stat().st_mtime for f in Path(settings.CORPUS_CONFIG_DIR, "attributes").glob("*/*.yaml")}
    return corpora, modes, presets


async def setup_cache(cache: Memcached) -> bool:
    """Setup disk cache and Memcached if needed.

    Args:
        cache: Memcached client.

    Returns:
        True if any action was needed (cache dir created or Memcached initialized), False otherwise.

    Raises:
        ValueError: If cache directory is set to the current working directory.
    """
    action_needed = False

    # Make sure that the cache dir is not the current working directory
    if settings.CACHE_DIR and Path.cwd() == settings.CACHE_DIR:
        raise ValueError("Cache directory cannot be the current working directory.")

    # Set up Memcached if needed
    if settings.MEMCACHED_SERVER and not await cache.get("multi:version"):
        memcached_data = {}
        version = uuid.uuid4().int
        corpora = get_corpus_timestamps()
        corpora_configs, config_modes, config_presets = get_corpus_config_timestamps()
        memcached_data["multi:version"] = version
        memcached_data["multi:version_config"] = version
        memcached_data["multi:corpora"] = set(corpora.keys())
        memcached_data["multi:config_corpora"] = set(corpora_configs.keys())
        memcached_data["multi:config_modes"] = config_modes
        memcached_data["multi:config_presets"] = config_presets
        for corpus in corpora:
            memcached_data[f"{corpus}:version"] = version
            memcached_data[f"{corpus}:last_update"] = corpora[corpus]
        for corpus in corpora_configs:
            memcached_data[f"{corpus}:version_config"] = version
            memcached_data[f"{corpus}:last_update_config"] = corpora_configs[corpus]
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
    # Evicted metadata must never make an old result namespace reachable again.
    for key in corpus_keys.values():
        if key not in versions:
            version = uuid.uuid4().int
            if await cache.add(key, version):
                versions[key] = version
            else:
                versions[key] = await cache.get(key, version)

    if single:
        return f"{corpus[0]}:{versions[corpus_keys[corpus[0]]]}"
    return {c: f"{c}:{versions[corpus_keys[c]]}" for c in corpus}


@overload
def cache_prefix_sync(
    cache: MemcachedSyncClient,
    corpus: str = "multi",
    config: bool = False,
) -> str: ...


@overload
def cache_prefix_sync(
    cache: MemcachedSyncClient,
    corpus: list[str],
    config: bool = False,
) -> dict[str, str]: ...


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
    for key in corpus_keys.values():
        if key not in versions:
            version = uuid.uuid4().int
            if cache.add(key, version):
                versions[key] = version
            else:
                versions[key] = cache.get(key, version)

    if single:
        return f"{corpus[0]}:{versions[corpus_keys[corpus[0]]]}"
    return {c: f"{c}:{versions[corpus_keys[c]]}" for c in corpus}

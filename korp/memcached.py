"""Memcached client with async and sync interfaces."""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from logging import getLogger
from typing import Any

import aiomcache
import anyio.from_thread

logger = getLogger("__name__")


class CacheError(Exception):
    """Memcached operation failed."""


def _encode_key(key: str) -> bytes:
    return key.encode("utf-8")


def _serialize(value: Any) -> bytes:
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize(raw: bytes) -> Any:
    return pickle.loads(raw)


def _parse_server(server: str) -> tuple[str, int]:
    if server.startswith("/"):
        raise ValueError("Unix sockets are not supported.")
    if ":" in server:
        host, port = server.rsplit(":", 1)
        return host, int(port)
    return server, 11211


@dataclass
class MemcachedSyncClient:
    """Synchronous Memcached client wrapper.

    Use `Memcached.get_client()` context manager to obtain an instance.
    """

    cache: Memcached

    def get(self, key: str, default: Any | None = None) -> Any | None:
        """Get a value from the cache by key.

        Args:
            key: The cache key.
            default: The default value to return if the key is not found.

        Returns:
            The cached value, or the default if not found.
        """
        return anyio.from_thread.run(self.cache.get, key, default)

    def add(self, key: str, value: Any) -> bool:
        """Add a value to the cache if the key does not already exist.

        Args:
            key: The cache key.
            value: The value to cache.

        Returns:
            True if the value was added, False if the key already exists.
        """
        return anyio.from_thread.run(self.cache.add, key, value)

    def set(self, key: str, value: Any) -> bool:
        """Set a value in the cache, overwriting any existing value.

        Args:
            key: The cache key.
            value: The value to cache.

        Returns:
            True if the value was set successfully.
        """
        return anyio.from_thread.run(self.cache.set, key, value)

    def get_many(self, keys: Iterable[str]) -> dict[str, Any]:
        """Get multiple values from the cache by their keys.

        Args:
            keys: An iterable of cache keys.

        Returns:
            A dictionary mapping keys to their cached values.
        """
        return anyio.from_thread.run(self.cache.get_many, keys)

    def set_many(self, items: Mapping[str, Any]) -> None:
        """Set multiple values in the cache.

        Args:
            items: A mapping of cache keys to values.
        """
        return anyio.from_thread.run(self.cache.set_many, items)


class Memcached:
    """Memcached client with async and sync interfaces."""

    def __init__(self) -> None:
        """Initialize the Memcached client."""
        self._client: aiomcache.Client | None = None
        self.active = False

    async def init(self, server: str | None) -> None:
        """Initialize the Memcached client with the given server address."""
        if not server:
            self.active = False
            return
        try:
            host, port = _parse_server(server)
        except ValueError as exc:
            logger.warning("Memcached disabled: %s", exc)
            self.active = False
            return

        client = aiomcache.Client(host, port)
        try:
            await client.get(b"__korp_ping__")
        except Exception as exc:
            await client.close()
            logger.warning("Could not connect to Memcached. Caching will be disabled. (%s)", exc)
            self.active = False
            return

        self._client = client
        self.active = True

    async def close(self) -> None:
        """Close the Memcached client."""
        if self._client is None:
            return
        await self._client.close()
        self._client = None
        self.active = False

    def _require_client(self) -> aiomcache.Client | None:
        if not self.active or self._client is None:
            return None
        return self._client

    async def get(self, key: str, default: Any | None = None) -> Any | None:
        """Get a value from the cache by key.

        Args:
            key: The cache key.
            default: The default value to return if the key is not found.

        Returns:
            The cached value, or the default if not found.

        Raises:
            CacheError: If there was an error accessing the cache.
        """
        client = self._require_client()
        if client is None:
            return default
        try:
            raw = await client.get(_encode_key(key))
        except Exception as exc:
            raise CacheError(str(exc)) from exc
        if raw is None:
            return default
        return _deserialize(raw)

    async def add(self, key: str, value: Any) -> bool:
        """Add a value to the cache if the key does not already exist.

        Args:
            key: The cache key.
            value: The value to cache.

        Returns:
            True if the value was added, False if the key already exists.

        Raises:
            CacheError: If there was an error accessing the cache.
        """
        client = self._require_client()
        if client is None:
            return False
        try:
            return await client.add(_encode_key(key), _serialize(value))
        except Exception as exc:
            raise CacheError(str(exc)) from exc

    async def set(self, key: str, value: Any) -> bool:
        """Set a value in the cache, overwriting any existing value.

        Args:
            key: The cache key.
            value: The value to cache.

        Returns:
            True if the value was set successfully.

        Raises:
            CacheError: If there was an error accessing the cache.
        """
        client = self._require_client()
        if client is None:
            return False
        try:
            return await client.set(_encode_key(key), _serialize(value))
        except Exception as exc:
            raise CacheError(str(exc)) from exc

    async def get_many(self, keys: Iterable[str]) -> dict[str, Any]:
        """Get multiple values from the cache by their keys.

        Args:
            keys: An iterable of cache keys.

        Returns:
            A dictionary mapping keys to their cached values.

        Raises:
            CacheError: If there was an error accessing the cache.
        """
        client = self._require_client()
        keys_list = list(keys)
        if client is None or not keys_list:
            return {}
        encoded = [_encode_key(k) for k in keys_list]
        try:
            values = await client.multi_get(*encoded)
        except Exception as exc:
            raise CacheError(str(exc)) from exc
        result = {}
        for key, raw in zip(keys_list, values, strict=True):
            if raw is None:
                continue
            result[key] = _deserialize(raw)
        return result

    async def set_many(self, items: Mapping[str, Any]) -> None:
        """Set multiple values in the cache.

        Args:
            items: A mapping of cache keys to values.

        Raises:
            CacheError: If there was an error accessing the cache.
        """
        client = self._require_client()
        if client is None or not items:
            return
        encoded = {_encode_key(k): _serialize(v) for k, v in items.items()}
        try:
            for key, value in encoded.items():
                await client.set(key, value)
        except Exception as exc:
            raise CacheError(str(exc)) from exc

    @contextmanager
    def get_client(self) -> Iterator[MemcachedSyncClient]:
        """Get a synchronous Memcached client wrapper.

        Yields:
            A synchronous cache client.
        """
        yield MemcachedSyncClient(self)


memcached = Memcached()

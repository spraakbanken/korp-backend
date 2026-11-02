"""Utilities used in pytest tests for Korp for testing with caching enabled."""


class FakeAiomcacheClient:
    """An in-memory stand-in for the `aiomcache.Client` methods that `korp.memcached.Memcached` uses.

    The values are stored as the bytes that `Memcached` serializes, as with a real Memcached server.
    """

    def __init__(self, host: str, port: int) -> None:  # ruff: ignore[unused-method-argument]
        """Initialize an empty cache; `host` and `port` are ignored."""
        self.data: dict[bytes, bytes] = {}

    async def get(self, key: bytes) -> bytes | None:
        """Return the value of `key`, or None if it is not cached."""
        return self.data.get(key)

    async def multi_get(self, *keys: bytes) -> tuple[bytes | None, ...]:
        """Return the values of `keys`, None for those not cached."""
        return tuple(self.data.get(key) for key in keys)

    async def add(self, key: bytes, value: bytes) -> bool:
        """Cache `value` as `key` unless `key` is already cached.

        Returns:
            Whether `value` was cached.
        """
        if key in self.data:
            return False
        self.data[key] = value
        return True

    async def set(self, key: bytes, value: bytes) -> bool:
        """Cache `value` as `key`.

        Returns:
            True.
        """
        self.data[key] = value
        return True

    async def close(self) -> None:
        """Do nothing."""

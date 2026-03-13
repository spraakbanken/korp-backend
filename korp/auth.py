"""Authorization for the Korp API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from korp import caching
from korp.dependencies import AuthContext, Ctx

if TYPE_CHECKING:
    from korp.cwb import CWB
    from korp.memcached import Memcached


class KorpAuthorizationError(Exception):
    """Custom exception for Korp authorization errors."""


@dataclass(frozen=True)
class ProtectionInfo:
    """Protection metadata for a corpus.

    Attributes:
        protected: Whether the corpus is access-restricted.
        details: Optional source-specific metadata (e.g., license type).
    """

    protected: bool
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_cache_value(cls, value: Any) -> ProtectionInfo | None:
        """Decode cache value to a `ProtectionInfo` object.

        Returns:
            Parsed `ProtectionInfo`, or `None` if the value is invalid.
        """
        if not isinstance(value, dict):
            return None
        protected = value.get("protected")
        if not isinstance(protected, bool):
            return None
        details = value.get("details", {})
        if not isinstance(details, dict):
            return None
        return cls(protected=protected, details=details)

    def to_cache_value(self) -> dict[str, Any]:
        """Encode ProtectionInfo to cache-friendly dictionary.

        Returns:
            A serializable mapping with `protected` and `details` keys.
        """
        return {"protected": self.protected, "details": dict(self.details)}


def _make_auth_context(ctx: Ctx) -> AuthContext:
    """Create an AuthContext from a request context.

    Returns:
        An AuthContext with request and cache settings from the given context.
    """
    return AuthContext(request=ctx.request, cache_enabled=ctx.common.cache)


async def get_protected_corpora(ctx: Ctx) -> list[str]:
    """Return a list of corpora with restricted access."""
    authorizer = ctx.request.app.state.authorizer
    if authorizer:
        return await authorizer.get_protected_corpora(_make_auth_context(ctx))
    return []


async def check_authorization(corpora: Iterable[str], ctx: Ctx) -> None:
    """Take a list of corpora, and if any of them are protected, check authorization.

    Args:
        corpora: List of corpus names.
        ctx: Request context used to build authorizer context.

    Raises:
        KorpAuthorizationError: If the user is not authorized to access one or more of the specified corpora.
    """
    authorizer = ctx.request.app.state.authorizer
    if authorizer:
        # Split parallel corpora
        corpora = [cc for c in corpora for cc in c.split("|")]

        success, unauthorized, message = await authorizer.check_authorization(corpora, _make_auth_context(ctx))
        if not success:
            if not message:
                message = "You do not have access to the following corpora: {}".format(", ".join(unauthorized))
            raise KorpAuthorizationError(message)


class Authorizer(ABC):
    """Base class for authorizer plugins.

    Plugin authors subclass this class to define:
    1. How corpus protection metadata is fetched.
    2. How request authorization is decided for protected corpora.

    Required module contract:
    - Your plugin module must export `AUTHORIZER_CLASS = YourAuthorizerSubclass`.

    Required methods to implement:
    - `_fetch_protection_info(corpora, auth_ctx)`:
      Fetch protection metadata for the provided corpus ids from your source (for example CWB info, a DB table, or an
      external API). This is an implementation hook and should not be called directly from plugin logic; use
      `_get_protection_info()` instead to ensure caching is used.
    - `get_protected_corpora(auth_ctx)`: Return all protected corpora (uppercase corpus ids). This is used by `/info`
      and similar "list all protected corpora" use cases.
    - `check_authorization(corpora, auth_ctx)`: Return `(success, unauthorized, message)` for the requested corpora.
      `unauthorized` should contain corpus ids that failed authorization.

    Protection metadata model:
    - Use `ProtectionInfo` for each corpus.
    - `ProtectionInfo.protected` is the canonical "restricted access" flag.
    - `ProtectionInfo.details` may contain plugin-specific metadata (for example license or policy attributes).

    Helper methods provided by this base class:
    - `_get_protection_info(corpora, auth_ctx)`:
      Main helper; resolves protection metadata via cache + fetch fallback. This is the method plugin code should call.
    - `_get_cached_protection_info(corpora, auth_ctx)`:
      Read plugin-scoped per-corpus protection metadata from Memcached. Called by `_get_protection_info()`.
    - `_store_protection_info(data, auth_ctx)`: Store plugin-scoped per-corpus protection metadata in Memcached. Called
      by `_get_protection_info()` after fetching missing entries.

    Recommended implementation pattern:
    - In `check_authorization`, call `_get_protection_info()` only for the requested corpora, then authorize only those
      where `ProtectionInfo.protected` is `True`.
    - In `get_protected_corpora`, list all corpora from your source of truth, resolve protection via
      `_get_protection_info()`, and return protected ids.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize subclass and set up plugin-scoped cache key suffix."""
        super().__init_subclass__(**kwargs)
        cls._protection_cache_key_suffix = f"auth_protection:{cls.__module__}.{cls.__name__}"

    def __init__(self, cwb: CWB, cache: Memcached) -> None:
        """Initialize authorizer with app-scoped dependencies."""
        self.cwb: CWB = cwb
        self.cache: Memcached = cache

    async def _get_cached_protection_info(
        self, corpora: list[str], auth_ctx: AuthContext
    ) -> tuple[dict[str, ProtectionInfo], list[str]]:
        """Load per-corpus protection metadata from cache.

        Args:
            corpora: Corpora to resolve protection metadata for.
            auth_ctx: Request-scoped authentication context.

        Returns:
            Tuple of:
                - resolved protection info by corpus
                - corpora missing from cache
        """
        if not auth_ctx.cache_enabled or not corpora:
            return {}, corpora

        prefixes = await caching.cache_prefix(self.cache, corpora)
        cache_keys = {f"{prefixes[corpus]}:{self._protection_cache_key_suffix}": corpus for corpus in corpora}
        cached_values = await self.cache.get_many(cache_keys.keys())

        resolved: dict[str, ProtectionInfo] = {}
        missing_corpora: list[str] = []

        for cache_key, corpus in cache_keys.items():
            info = ProtectionInfo.from_cache_value(cached_values.get(cache_key))
            if info is None:
                missing_corpora.append(corpus)
            else:
                resolved[corpus] = info

        return resolved, missing_corpora

    async def _store_protection_info(self, data: dict[str, ProtectionInfo], auth_ctx: AuthContext) -> None:
        """Store per-corpus protection metadata in cache."""
        if not auth_ctx.cache_enabled or not data:
            return

        prefixes = await caching.cache_prefix(self.cache, list(data.keys()))
        cache_data = {
            f"{prefixes[corpus]}:{self._protection_cache_key_suffix}": info.to_cache_value()
            for corpus, info in data.items()
        }
        await self.cache.set_many(cache_data)

    async def _get_protection_info(self, corpora: list[str], auth_ctx: AuthContext) -> dict[str, ProtectionInfo]:
        """Resolve protection metadata for corpora using cache + source fetch.

        This method first checks plugin-scoped cached metadata, then asks the plugin to fetch missing entries, and
        finally stores fetched entries in cache.

        Returns:
            Protection metadata keyed by corpus id.
        """
        if not corpora:
            return {}

        resolved, missing = await self._get_cached_protection_info(corpora, auth_ctx)

        if missing:
            fetched = await self._fetch_protection_info(missing, auth_ctx)
            # Missing keys from fetch are treated as "not protected"
            for corpus in missing:
                fetched.setdefault(corpus, ProtectionInfo(protected=False))
            await self._store_protection_info(fetched, auth_ctx)
            resolved.update(fetched)

        return {corpus: resolved.get(corpus, ProtectionInfo(protected=False)) for corpus in corpora}

    @abstractmethod
    async def _fetch_protection_info(self, corpora: list[str], auth_ctx: AuthContext) -> dict[str, ProtectionInfo]:
        """Fetch per-corpus protection metadata from the plugin-specific source.

        This is a low-level subclass hook. Do not call this directly in plugin code; call `_get_protection_info()` to
        ensure caching is used.
        """

    @abstractmethod
    async def get_protected_corpora(self, auth_ctx: AuthContext) -> list[str]:
        """Get list of corpora with restricted access, in uppercase."""

    @abstractmethod
    async def check_authorization(
        self, corpora: list[str], auth_ctx: AuthContext
    ) -> tuple[bool, list[str], str | None]:
        """Take a list of corpora and check that the user has permission to access them.

        Returns:
            A tuple containing:
                - A boolean indicating if access is granted.
                - A list of unauthorized corpora (if access is denied).
                - An optional message for the user. If `None`, a default message including a list of unauthorized
                  corpora will be used.
        """

"""Helpers for CWB-based corpus protection metadata."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import TYPE_CHECKING, Any

import anyio.to_thread

from korp import auth, caching, cqp, utils

if TYPE_CHECKING:
    from korp.cwb import CWB
    from korp.dependencies import AuthContext
    from korp.memcached import Memcached


def list_corpora(cwb: CWB) -> list[str]:
    """Return all corpora reported by CQP."""
    corpora_lines = cwb.run_cqp("show corpora;")
    next(corpora_lines, None)  # Skip CQP version
    return [utils.normalize_corpus_id(corpus) for corpus in corpora_lines]


async def get_protected_corpora(
    cwb: CWB, get_protection_info: Callable[[list[str]], Awaitable[dict[str, auth.ProtectionInfo]]]
) -> list[str]:
    """Return the protected corpora among all corpora reported by CQP.

    Undefined corpora, which CQP cannot select, are listed as protected, as their protection status cannot be read;
    only a status cached before a corpus became undefined may be used instead. `/info` caches its result, so listing
    them as public could show a protected corpus as public after its data is repaired without changing its registry
    file; this way, a repaired public corpus is instead listed as protected until the cache is invalidated.

    Args:
        cwb: CWB interface.
        get_protection_info: Function returning protection metadata keyed by corpus for the given corpora, such as
            `Authorizer._get_protection_info` with the authentication context bound. It should raise `CQPError` if a
            corpus is undefined; it is then called again without the undefined corpora.

    Returns:
        Lowercase ids of the protected corpora.

    Raises:
        CQPError: If CQP reports an error that is not caused by an undefined corpus.
    """
    corpora = list_corpora(cwb)
    try:
        protection_info = await get_protection_info(corpora)
    except cqp.CQPError:
        # The error message names at most one undefined corpus, and it may not even show that the error is caused by
        # one, as CWB may print its own error first, so check all the corpora
        undefined = set(await anyio.to_thread.run_sync(cwb.find_undefined_corpora, corpora))
        if not undefined:
            raise
        protection_info = await get_protection_info([corpus for corpus in corpora if corpus not in undefined])
        # Not stored in the protection cache, so that the corpora are checked again once their data is repaired
        protection_info.update(dict.fromkeys(undefined, auth.ProtectionInfo(protected=True)))
    return [corpus for corpus in corpora if protection_info[corpus].protected]


def _normalize_detail_keys(detail_keys: Iterable[str] | None) -> set[str]:
    """Normalize optional detail keys to a lowercase set.

    Returns:
        Lowercase allowlist of detail keys, excluding `Protected`.
    """
    if detail_keys is None:
        return set()
    return {
        key.casefold() for key in detail_keys if isinstance(key, str) and key.strip() and key.casefold() != "protected"
    }


def _extract_details(info: dict[str, Any], normalized_detail_keys: set[str]) -> dict[str, Any]:
    """Extract whitelisted detail keys from a CWB info mapping.

    Returns:
        Mapping containing only whitelisted detail keys.
    """
    if not normalized_detail_keys:
        return {}
    return {
        key: value for key, value in info.items() if isinstance(key, str) and key.casefold() in normalized_detail_keys
    }


def _parse_protection_info_from_cached_corpus_info(
    cached_corpus_info: Any,
    normalized_detail_keys: set[str],
) -> auth.ProtectionInfo | None:
    """Parse protection metadata from `get_corpus_info` cache shape.

    Returns:
        Parsed protection metadata, or `None` if cache data is invalid.
    """
    if not isinstance(cached_corpus_info, dict):
        return None
    info = cached_corpus_info.get("info")
    if not isinstance(info, dict):
        return None
    protected = False
    for key, value in info.items():
        if isinstance(key, str) and key.casefold() == "protected":
            protected = str(value).lower() == "true"
            break
    details = _extract_details(info, normalized_detail_keys)
    return auth.ProtectionInfo(protected=protected, details=details)


async def fetch_protection_info(
    cwb: CWB,
    corpora: list[str],
    cache: Memcached,
    auth_ctx: AuthContext,
    detail_keys: Iterable[str] | None = None,
) -> dict[str, auth.ProtectionInfo]:
    """Fetch protection metadata from CWB, reusing cached corpus info when available.

    Args:
        cwb: CWB interface.
        corpora: Corpora to resolve protection metadata for.
        cache: Memcached client.
        auth_ctx: Authentication context.
        detail_keys: Optional list of CWB info keys to include in `ProtectionInfo.details`.

    Returns:
        Protection metadata keyed by corpus.
    """
    if not corpora:
        return {}

    result: dict[str, auth.ProtectionInfo] = {}
    missing = corpora
    normalized_detail_keys = _normalize_detail_keys(detail_keys)

    if auth_ctx.cache_enabled:
        prefixes = await caching.cache_prefix(cache, corpora)
        info_cache_keys = {f"{prefixes[corpus]}:info": corpus for corpus in corpora}
        cached_corpora = await cache.get_many(info_cache_keys.keys())

        missing = []
        for info_cache_key, corpus in info_cache_keys.items():
            protection = _parse_protection_info_from_cached_corpus_info(
                cached_corpora.get(info_cache_key),
                normalized_detail_keys,
            )
            if protection is None:
                missing.append(corpus)
            else:
                result[corpus] = protection

    if missing:
        cmd = []
        for corpus in missing:
            cmd += [f"{corpus.upper()};", "info; .EOL.;"]
        cmd += ["exit;"]

        lines = cwb.run_cqp(cmd)
        next(lines, None)  # Skip CQP version

        for corpus in missing:
            is_protected = False
            details: dict[str, Any] = {}
            for line in lines:
                if line == cqp.END_OF_LINE:
                    break
                if ":" in line and not line.endswith(":"):
                    key, value = (part.strip() for part in line.split(":", 1))
                    if key.casefold() == "protected":
                        is_protected = value.lower() == "true"
                    elif key.casefold() in normalized_detail_keys:
                        details[key] = value
            result[corpus] = auth.ProtectionInfo(protected=is_protected, details=details)

    return result

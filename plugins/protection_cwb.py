"""Helpers for CWB-based corpus protection metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from korp import auth, caching, cqp

if TYPE_CHECKING:
    from korp.cwb import CWB
    from korp.dependencies import AuthContext
    from korp.memcached import Memcached


def list_corpora(cwb: CWB) -> list[str]:
    """Return all corpora reported by CQP."""
    corpora_lines = cwb.run_cqp("show corpora;")
    next(corpora_lines, None)  # Skip CQP version
    return list(corpora_lines)


def _parse_protection_info_from_cached_corpus_info(cached_corpus_info: Any) -> auth.ProtectionInfo | None:
    """Parse protection metadata from `get_corpus_info` cache shape.

    Returns:
        Parsed protection metadata, or `None` if cache data is invalid.
    """
    if not isinstance(cached_corpus_info, dict):
        return None
    info = cached_corpus_info.get("info")
    if not isinstance(info, dict):
        return None
    protected = info.get("Protected")
    if protected is None:
        return auth.ProtectionInfo(protected=False)
    return auth.ProtectionInfo(protected=str(protected).lower() == "true")


async def fetch_protection_info(
    cwb: CWB,
    corpora: list[str],
    cache: Memcached,
    auth_ctx: AuthContext,
) -> dict[str, auth.ProtectionInfo]:
    """Fetch protection metadata from CWB, reusing cached corpus info when available.

    Returns:
        Protection metadata keyed by corpus.
    """
    if not corpora:
        return {}

    result: dict[str, auth.ProtectionInfo] = {}
    missing = corpora

    if auth_ctx.cache_enabled:
        prefixes = await caching.cache_prefix(cache, corpora)
        info_cache_keys = {f"{prefixes[corpus]}:info": corpus for corpus in corpora}
        cached_corpora = await cache.get_many(info_cache_keys.keys())

        missing = []
        for info_cache_key, corpus in info_cache_keys.items():
            protection = _parse_protection_info_from_cached_corpus_info(cached_corpora.get(info_cache_key))
            if protection is None:
                missing.append(corpus)
            else:
                result[corpus] = protection

    if missing:
        cmd = []
        for corpus in missing:
            cmd += [f"{corpus};", "info; .EOL.;"]
        cmd += ["exit;"]

        lines = cwb.run_cqp(cmd)
        next(lines, None)  # Skip CQP version

        for corpus in missing:
            is_protected = False
            for line in lines:
                if line == cqp.END_OF_LINE:
                    break
                if ":" in line and not line.endswith(":"):
                    key, value = (part.strip() for part in line.split(":", 1))
                    if key == "Protected":
                        is_protected = value.lower() == "true"
            result[corpus] = auth.ProtectionInfo(protected=is_protected)

    return result

"""Unit tests for protected-corpora lookup in auth plugins."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import anyio
import pytest

from korp import cqp
from plugins.auth import Auth
from plugins.auth_jwt import AuthJWT


class FakeCWB:
    """Minimal CWB stub for testing batched protected-corpora reads."""

    def __init__(
        self,
        corpora: list[str],
        protected: dict[str, bool],
        details: dict[str, dict[str, str]] | None = None,
    ) -> None:
        """Initialize fake corpora and protected flags."""
        self.corpora = corpora
        self.protected = protected
        self.details = details or {}
        self.calls: list[str | list[str]] = []

    def run_cqp(
        self, command: str | list[str], attr_ignore: bool = False, abort_signal: Any | None = None
    ) -> Iterator[str]:
        """Return canned CQP output and record commands."""
        del attr_ignore, abort_signal
        self.calls.append(command)

        if command == "show corpora;":
            return iter(["3.5.0", *self.corpora])

        expected_batch_cmd: list[str] = []
        for corpus in self.corpora:
            expected_batch_cmd += [f"{corpus};", "info; .EOL.;"]
        expected_batch_cmd += ["exit;"]
        assert command == expected_batch_cmd

        lines = ["3.5.0"]
        for corpus in self.corpora:
            lines += [f"Name: {corpus}"]
            if corpus in self.protected:
                value = "true" if self.protected[corpus] else "false"
                lines += [f"Protected: {value}"]
            for key, value in self.details.get(corpus, {}).items():
                lines += [f"{key}: {value}"]
            lines += [cqp.END_OF_LINE]
        return iter(lines)


class FakeCache:
    """Minimal async cache stub used by auth plugins."""

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        """Initialize cache with optional pre-populated key/value pairs."""
        self.values = values or {}
        self.set_many_calls: list[dict[str, Any]] = []

    async def get(self, key: str) -> Any | None:
        """Return cached value for key, or `None` if missing."""
        return self.values.get(key)

    async def add(self, key: str, value: Any) -> bool:
        """Pretend cache add always succeeds.

        Returns:
            Always `True`.
        """
        self.values[key] = value
        return True

    async def get_many(self, keys: Any) -> dict[str, Any]:
        """Return all matching cached values for requested keys."""
        key_list = list(keys)
        return {k: self.values[k] for k in key_list if k in self.values}

    async def set_many(self, items: dict[str, Any]) -> None:
        """Store multiple key/value pairs."""
        self.values.update(items)
        self.set_many_calls.append(items)


@pytest.mark.parametrize("authorizer_cls", [Auth, AuthJWT])
def test_get_protected_corpora_uses_batched_cwb_info_calls(authorizer_cls: type[Auth] | type[AuthJWT]) -> None:
    """Read corpus info in one CQP invocation and return uppercase protected ids."""

    async def _run() -> None:
        cwb = FakeCWB(corpora=["alpha", "BETA", "gamma"], protected={"alpha": True, "BETA": False})
        authorizer = authorizer_cls(cwb=cwb, cache=FakeCache())  # pyright: ignore

        protected = await authorizer.get_protected_corpora(SimpleNamespace(cache_enabled=False))  # pyright: ignore

        assert protected == ["ALPHA"]
        assert cwb.calls == [
            "show corpora;",
            ["alpha;", "info; .EOL.;", "BETA;", "info; .EOL.;", "gamma;", "info; .EOL.;", "exit;"],
        ]

    anyio.run(_run)


@pytest.mark.parametrize("authorizer_cls", [Auth, AuthJWT])
def test_get_protected_corpora_skips_info_call_when_no_corpora(authorizer_cls: type[Auth] | type[AuthJWT]) -> None:
    """Avoid running batch info command when CWB has no corpora."""

    async def _run() -> None:
        cwb = FakeCWB(corpora=[], protected={})
        authorizer = authorizer_cls(cwb=cwb, cache=FakeCache())  # pyright: ignore

        protected = await authorizer.get_protected_corpora(SimpleNamespace(cache_enabled=False))  # pyright: ignore

        assert protected == []
        assert cwb.calls == ["show corpora;"]

    anyio.run(_run)


@pytest.mark.parametrize("authorizer_cls", [Auth, AuthJWT])
def test_get_protected_corpora_reads_protected_flag_from_corpus_info_cache(
    authorizer_cls: type[Auth] | type[AuthJWT],
) -> None:
    """Use cached `<prefix>:info` values and avoid CQP info calls when all corpora are cached."""

    async def _run() -> None:
        cwb = FakeCWB(corpora=["alpha", "BETA"], protected={})
        cache = FakeCache(
            {
                "alpha:version": 7,
                "BETA:version": 11,
                "alpha:7:info": {"info": {"Protected": "true"}},
                "BETA:11:info": {"info": {"Protected": "false"}},
            }
        )
        authorizer = authorizer_cls(cwb=cwb, cache=cache)  # pyright: ignore

        protected = await authorizer.get_protected_corpora(SimpleNamespace(cache_enabled=True))  # pyright: ignore

        assert protected == ["ALPHA"]
        assert cwb.calls == ["show corpora;"]
        assert len(cache.set_many_calls) == 1

    anyio.run(_run)


@pytest.mark.parametrize("authorizer_cls", [Auth, AuthJWT])
def test_get_protected_corpora_uses_plugin_scoped_protection_cache(authorizer_cls: type[Auth] | type[AuthJWT]) -> None:
    """Read plugin-scoped protection cache without invoking CQP info."""

    async def _run() -> None:
        cwb = FakeCWB(corpora=["alpha", "BETA"], protected={})
        suffix = f"auth_protection:{authorizer_cls.__module__}.{authorizer_cls.__name__}"
        cache = FakeCache(
            {
                "alpha:version": 7,
                "BETA:version": 11,
                f"alpha:7:{suffix}": {"protected": True, "details": {"license": "a"}},
                f"BETA:11:{suffix}": {"protected": False, "details": {}},
            }
        )
        authorizer = authorizer_cls(cwb=cwb, cache=cache)  # pyright: ignore

        protected = await authorizer.get_protected_corpora(SimpleNamespace(cache_enabled=True))  # pyright: ignore

        assert protected == ["ALPHA"]
        assert cwb.calls == ["show corpora;"]
        assert cache.set_many_calls == []

    anyio.run(_run)

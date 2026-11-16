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


def test_builtin_authorizers_vary_private_cache_by_authorization_header() -> None:
    """Ensure built-in authorizers configure Vary headers for private cache."""
    assert Auth.cache_vary_headers() == ("Authorization",)
    assert AuthJWT.cache_vary_headers() == ("Authorization",)


class FakeCWB:
    """Minimal CWB stub for testing batched protected-corpora reads."""

    def __init__(
        self,
        corpora: list[str],
        protected: dict[str, bool],
        details: dict[str, dict[str, str]] | None = None,
        undefined: set[str] | None = None,
    ) -> None:
        """Initialize fake corpora, protected flags and undefined corpora.

        Selecting a corpus in `undefined` in CQP fails, like selecting a corpus without readable data.
        """
        self.corpora = corpora
        self.protected = protected
        self.details = details or {}
        self.undefined = {corpus.upper() for corpus in undefined or set()}
        self.calls: list[str | list[str]] = []

    def run_cqp(
        self, command: str | list[str], attr_ignore: bool = False, abort_signal: Any | None = None
    ) -> Iterator[str]:
        """Return canned CQP output and record commands.

        Like `CWB.run_cqp`, the returned iterator raises `CQPError` when consumed if the command selects an undefined
        corpus.
        """
        del attr_ignore, abort_signal
        self.calls.append(command)

        if command == "show corpora;":
            return iter(["3.5.0", *self.corpora])

        corpora_by_id = {corpus.upper(): corpus for corpus in self.corpora}
        selected = [select.removesuffix(";") for select in command[:-1:2]]
        expected_batch_cmd: list[str] = []
        for corpus_id in selected:
            expected_batch_cmd += [f"{corpus_id};", "info; .EOL.;"]
        expected_batch_cmd += ["exit;"]
        assert command == expected_batch_cmd
        assert all(corpus_id in corpora_by_id for corpus_id in selected)

        if undefined := [corpus_id for corpus_id in selected if corpus_id in self.undefined]:

            def _raise_error() -> Iterator[str]:
                raise cqp.CQPError(f"Corpus ``{undefined[0]}'' is undefined")
                yield  # pragma: no cover

            return _raise_error()

        lines = ["3.5.0"]
        for corpus in (corpora_by_id[corpus_id] for corpus_id in selected):
            lines += [f"Name: {corpus}"]
            if corpus in self.protected:
                value = "true" if self.protected[corpus] else "false"
                lines += [f"Protected: {value}"]
            for key, value in self.details.get(corpus, {}).items():
                lines += [f"{key}: {value}"]
            lines += [cqp.END_OF_LINE]
        return iter(lines)

    def find_undefined_corpora(self, corpora: list[str]) -> list[str]:
        """Return the undefined corpora in `corpora` and record the call.

        Returns:
            The undefined corpora, as given and in the same order.
        """
        self.calls.append(f"find_undefined_corpora {corpora}")
        return [corpus for corpus in corpora if corpus.upper() in self.undefined]


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
    """Read corpus info in one CQP invocation and return lowercase protected ids."""

    async def _run() -> None:
        cwb = FakeCWB(corpora=["alpha", "BETA", "gamma"], protected={"alpha": True, "BETA": False})
        authorizer = authorizer_cls(cwb=cwb, cache=FakeCache())  # pyright: ignore

        protected = await authorizer.get_protected_corpora(SimpleNamespace(cache_enabled=False))  # pyright: ignore

        assert protected == ["alpha"]
        assert cwb.calls == [
            "show corpora;",
            ["ALPHA;", "info; .EOL.;", "BETA;", "info; .EOL.;", "GAMMA;", "info; .EOL.;", "exit;"],
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
                "beta:version": 11,
                "alpha:7:info": {"info": {"Protected": "true"}},
                "beta:11:info": {"info": {"Protected": "false"}},
            }
        )
        authorizer = authorizer_cls(cwb=cwb, cache=cache)  # pyright: ignore

        protected = await authorizer.get_protected_corpora(SimpleNamespace(cache_enabled=True))  # pyright: ignore

        assert protected == ["alpha"]
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
                "beta:version": 11,
                f"alpha:7:{suffix}": {"protected": True, "details": {"license": "a"}},
                f"beta:11:{suffix}": {"protected": False, "details": {}},
            }
        )
        authorizer = authorizer_cls(cwb=cwb, cache=cache)  # pyright: ignore

        protected = await authorizer.get_protected_corpora(SimpleNamespace(cache_enabled=True))  # pyright: ignore

        assert protected == ["alpha"]
        assert cwb.calls == ["show corpora;"]
        assert cache.set_many_calls == []

    anyio.run(_run)


@pytest.mark.parametrize("authorizer_cls", [Auth, AuthJWT])
def test_get_protected_corpora_lists_undefined_corpora_as_protected(
    authorizer_cls: type[Auth] | type[AuthJWT],
) -> None:
    """List undefined corpora as protected, reading the info of the other corpora after checking for them."""

    async def _run() -> None:
        cwb = FakeCWB(
            corpora=["alpha", "beta", "gamma", "delta"],
            protected={"alpha": True, "beta": False, "gamma": False},
            undefined={"gamma", "delta"},
        )
        authorizer = authorizer_cls(cwb=cwb, cache=FakeCache())  # pyright: ignore

        protected = await authorizer.get_protected_corpora(SimpleNamespace(cache_enabled=False))  # pyright: ignore

        assert protected == ["alpha", "gamma", "delta"]
        assert cwb.calls == [
            "show corpora;",
            [
                *("ALPHA;", "info; .EOL.;", "BETA;", "info; .EOL.;"),
                *("GAMMA;", "info; .EOL.;", "DELTA;", "info; .EOL.;", "exit;"),
            ],
            "find_undefined_corpora ['alpha', 'beta', 'gamma', 'delta']",
            ["ALPHA;", "info; .EOL.;", "BETA;", "info; .EOL.;", "exit;"],
        ]

    anyio.run(_run)


@pytest.mark.parametrize("authorizer_cls", [Auth, AuthJWT])
def test_get_protected_corpora_does_not_cache_undefined_corpora(authorizer_cls: type[Auth] | type[AuthJWT]) -> None:
    """Cache the protection of defined corpora only, so that undefined ones are checked again on the next request."""

    async def _run() -> None:
        cwb = FakeCWB(corpora=["alpha", "beta"], protected={"alpha": True, "beta": False}, undefined={"beta"})
        suffix = f"auth_protection:{authorizer_cls.__module__}.{authorizer_cls.__name__}"
        cache = FakeCache({"alpha:version": 7, "beta:version": 11})
        authorizer = authorizer_cls(cwb=cwb, cache=cache)  # pyright: ignore
        auth_ctx = SimpleNamespace(cache_enabled=True)

        assert await authorizer.get_protected_corpora(auth_ctx) == ["alpha", "beta"]  # pyright: ignore
        assert cache.set_many_calls == [{f"alpha:7:{suffix}": {"protected": True, "details": {}}}]

        cwb.calls.clear()
        assert await authorizer.get_protected_corpora(auth_ctx) == ["alpha", "beta"]  # pyright: ignore
        assert cwb.calls == [
            "show corpora;",
            ["BETA;", "info; .EOL.;", "exit;"],
            "find_undefined_corpora ['alpha', 'beta']",
        ]
        assert len(cache.set_many_calls) == 1

    anyio.run(_run)


class FakeCWBWithOtherError(FakeCWB):
    """CWB stub for which selecting an undefined corpus fails, but `find_undefined_corpora` does not report it."""

    def find_undefined_corpora(self, corpora: list[str]) -> list[str]:
        """Record the call and return no undefined corpora.

        Returns:
            An empty list.
        """
        super().find_undefined_corpora(corpora)
        return []


@pytest.mark.parametrize("authorizer_cls", [Auth, AuthJWT])
def test_get_protected_corpora_raises_cqp_error_without_undefined_corpora(
    authorizer_cls: type[Auth] | type[AuthJWT],
) -> None:
    """Re-raise a CQP error that is not caused by an undefined corpus."""

    async def _run() -> None:
        cwb = FakeCWBWithOtherError(corpora=["alpha", "beta"], protected={}, undefined={"beta"})
        authorizer = authorizer_cls(cwb=cwb, cache=FakeCache())  # pyright: ignore

        with pytest.raises(cqp.CQPError):
            await authorizer.get_protected_corpora(SimpleNamespace(cache_enabled=False))  # pyright: ignore
        assert cwb.calls[-1] == "find_undefined_corpora ['alpha', 'beta']"

    anyio.run(_run)


@pytest.mark.parametrize("authorizer_cls", [Auth, AuthJWT])
def test_check_authorization_fails_on_undefined_corpora(authorizer_cls: type[Auth] | type[AuthJWT]) -> None:
    """Fail closed with a CQP error for an undefined corpus, whose protection cannot be read."""

    async def _run() -> None:
        cwb = FakeCWB(corpora=["alpha", "beta"], protected={"alpha": False}, undefined={"beta"})
        authorizer = authorizer_cls(cwb=cwb, cache=FakeCache())  # pyright: ignore
        auth_ctx = SimpleNamespace(cache_enabled=False, request=SimpleNamespace(headers={}))

        with pytest.raises(cqp.CQPError):
            await authorizer.check_authorization(["alpha", "beta"], auth_ctx)  # pyright: ignore
        assert not any(isinstance(call, str) and call.startswith("find_undefined_corpora") for call in cwb.calls)

    anyio.run(_run)

"""Unit tests for checking undefined corpora in `korp.api.routers.info`."""

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any, cast

import anyio
import pytest

from korp.api.routers import info
from korp.api.routers.info import CorpusInfoRequest
from korp.cqp import CQPError
from korp.dependencies import CommonParams, Ctx


class FakeCache:
    """A dict-based stand-in for the `Memcached` methods used by `_split_undefined_corpora`."""

    def __init__(self) -> None:
        """Initialize an empty cache."""
        self.data: dict[str, Any] = {}

    async def get(self, key: str) -> Any | None:
        """Return the value of `key`, or None if it is not cached."""
        return self.data.get(key)

    async def get_many(self, keys: Iterable[str]) -> dict[str, Any]:
        """Return the cached items among `keys`."""
        return {key: self.data[key] for key in keys if key in self.data}

    async def add(self, key: str, value: Any) -> bool:
        """Cache `value` as `key` unless `key` is already cached.

        Returns:
            Whether `value` was cached.
        """
        if key in self.data:
            return False
        self.data[key] = value
        return True


class FakeCWB:
    """A stand-in for `CWB` that records the calls to `find_undefined_corpora`."""

    def __init__(self, undefined: frozenset[str] = frozenset(), unreadable: frozenset[str] = frozenset()) -> None:
        """Initialize with the corpora that are undefined.

        Args:
            undefined: The corpora that have no registry file.
            unreadable: The corpora that have a registry file but whose data cannot be read.
        """
        self.undefined = undefined
        self.unreadable = unreadable
        self.calls: list[tuple[list[str], bool]] = []

    def find_undefined_corpora(self, corpora: Iterable[str], select_in_cqp: bool = True) -> list[str]:
        """Record the call.

        Returns:
            The undefined corpora in `corpora`, including the unreadable ones if `select_in_cqp`.
        """
        corpora = list(corpora)
        self.calls.append((corpora, select_in_cqp))
        return [
            corpus for corpus in corpora if corpus in self.undefined or (select_in_cqp and corpus in self.unreadable)
        ]


def _split(
    cwb: FakeCWB, cache: FakeCache, corpora: list[str], *, cache_enabled: bool, select_in_cqp: bool = True
) -> Any:
    """Return the result of `_split_undefined_corpora` with a context of `cwb` and `cache`."""
    ctx = cast(Ctx, SimpleNamespace(common=CommonParams(cache=cache_enabled), cache=cache, cwb=cwb))
    return anyio.run(info._split_undefined_corpora, ctx, corpora, select_in_cqp)


@pytest.mark.parametrize("cache_enabled", [True, False])
def test_split_in_order(cache_enabled: bool) -> None:
    """The defined and undefined corpora are returned in the order of the argument."""
    cwb = FakeCWB(frozenset({"b", "d"}))

    assert _split(cwb, FakeCache(), ["d", "a", "b", "c"], cache_enabled=cache_enabled) == (["a", "c"], ["d", "b"])


def test_cached() -> None:
    """The result of the CQP check is cached for the same set of corpora, and returned in the argument order."""
    cwb = FakeCWB(frozenset({"b"}))
    cache = FakeCache()

    assert _split(cwb, cache, ["a", "b"], cache_enabled=True) == (["a"], ["b"])
    assert _split(cwb, cache, ["b", "a"], cache_enabled=True) == (["a"], ["b"])
    assert cwb.calls == [(["a", "b"], True)]


def test_registry_check_not_cached() -> None:
    """The result of checking only the registry files is not cached."""
    cwb = FakeCWB(frozenset({"b"}))
    cache = FakeCache()

    for _ in range(2):
        assert _split(cwb, cache, ["a", "b"], cache_enabled=True, select_in_cqp=False) == (["a"], ["b"])
    assert cwb.calls == [(["a", "b"], False), (["a", "b"], False)]
    assert cache.data == {}


def test_not_cached_if_cache_disabled() -> None:
    """The corpora are checked on every call if caching is disabled."""
    cwb = FakeCWB()
    cache = FakeCache()

    for _ in range(2):
        assert _split(cwb, cache, ["a"], cache_enabled=False) == (["a"], [])
    assert cwb.calls == [(["a"], True), (["a"], True)]
    assert cache.data == {}


class FakeGetCorpusInfo:
    """A stand-in for `get_corpus_info` that records its calls and fails like CQP for an unreadable corpus."""

    def __init__(
        self, cwb: FakeCWB, error: str | None = None, unreadable_error: str = "Corpus ``{corpus}'' is undefined"
    ) -> None:
        """Initialize with the corpora of `cwb`.

        Args:
            cwb: The `FakeCWB` whose undefined and unreadable corpora cause an error.
            error: An error message to raise always instead.
            unreadable_error: The error message for an undefined or unreadable corpus, with `{corpus}` replaced with
                the uppercase corpus id.
        """
        self.cwb = cwb
        self.error = error
        self.unreadable_error = unreadable_error
        self.calls: list[list[str]] = []

    async def __call__(self, ctx: Ctx, corpora: list[str]) -> dict[str, Any]:  # ruff: ignore[unused-method-argument]
        """Record the call.

        Returns:
            Information with an empty dict for each corpus.

        Raises:
            CQPError: If `error` is set or a corpus is unreadable or undefined.
        """
        self.calls.append(corpora)
        if self.error:
            raise CQPError(self.error)
        for corpus in corpora:
            if corpus in self.cwb.undefined or corpus in self.cwb.unreadable:
                raise CQPError(self.unreadable_error.format(corpus=corpus.upper()))
        return {"corpora": {corpus: {} for corpus in corpora}, "total_size": 0, "total_sentences": 0}


def _report(
    cwb: FakeCWB, corpora: list[str], **get_corpus_info_kwargs: str
) -> tuple[dict[str, Any], FakeGetCorpusInfo]:
    """Return the result of `/corpora/info` with `report_undefined_corpora` for `corpora`, without caching.

    Args:
        cwb: The `FakeCWB` to use.
        corpora: The requested corpora.
        get_corpus_info_kwargs: Keyword arguments to `FakeGetCorpusInfo`.

    Returns:
        The result and the `FakeGetCorpusInfo` used instead of `get_corpus_info`.
    """
    get_corpus_info = FakeGetCorpusInfo(cwb, **get_corpus_info_kwargs)
    ctx = cast(Ctx, SimpleNamespace(common=CommonParams(cache=False), cache=FakeCache(), cwb=cwb))
    request = CorpusInfoRequest(corpora=corpora, report_undefined_corpora=True)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(info, "get_corpus_info", get_corpus_info)
        result = anyio.run(info._get_requested_corpus_info, ctx, request)
    return result, get_corpus_info


def test_report_registry_check_only() -> None:
    """If getting the information succeeds after checking the registry files, CQP is not used for checking."""
    cwb = FakeCWB(frozenset({"b"}), frozenset({"x"}))

    result, get_corpus_info = _report(cwb, ["a", "b", "c"])

    assert list(result["corpora"]) == ["a", "c"]
    assert result["undefined_corpora"] == ["b"]
    assert cwb.calls == [(["a", "b", "c"], False)]
    assert get_corpus_info.calls == [["a", "c"]]


@pytest.mark.parametrize(
    "unreadable_error",
    ["Corpus ``{corpus}'' is undefined", "CL MemBlob:mmapfile(): Can't open file {corpus}/word.corpus"],
    ids=["undefined", "cl_error"],
)
def test_report_unreadable(unreadable_error: str) -> None:
    """If a corpus with a registry file is undefined in CQP, all are selected in CQP, and the information retried.

    The error message need not tell that the corpus is undefined, as CWB may print its own error before that. The
    undefined corpora are in the order of the request.
    """
    cwb = FakeCWB(frozenset({"c"}), frozenset({"a", "d"}))

    result, get_corpus_info = _report(cwb, ["a", "b", "c", "d"], unreadable_error=unreadable_error)

    assert list(result["corpora"]) == ["b"]
    assert result["undefined_corpora"] == ["a", "c", "d"]
    assert cwb.calls == [(["a", "b", "c", "d"], False), (["a", "b", "c", "d"], True)]
    assert get_corpus_info.calls == [["a", "b", "d"], ["b"]]


def test_report_other_error() -> None:
    """A CQP error is raised if selecting the corpora in CQP finds no other undefined corpora."""
    cwb = FakeCWB(frozenset({"c"}))

    with pytest.raises(CQPError, match=r"^Out of memory$"):
        _report(cwb, ["a", "b", "c"], error="Out of memory")
    assert cwb.calls == [(["a", "b", "c"], False), (["a", "b", "c"], True)]

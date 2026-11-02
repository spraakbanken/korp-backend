"""Pytest tests for the Korp `/info` and `/corpora/info` endpoints."""

import os
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from http import HTTPStatus
from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from korp.dependencies import AuthContext
from tests.corpusutils import CORPUS_WITHOUT_DATA, copy_corpus, write_broken_registry_file
from tests.testutils import get_response_json, validate_response_contract


@pytest.fixture
def corpus_info(get_json: Callable, corpora: list[str]) -> Callable[[list[str]], dict]:  # ruff: ignore[unused-function-argument]
    """Return function returning `/corpora/info` response for list of corpora."""

    def _corpus_info(corpus_list: list[str]) -> dict:
        """Return `/corpora/info` response for the corpora in `corpus_list`."""
        return get_json(
            "/corpora/info",
            params={
                "cache": "false",
                "corpora": ",".join(corpus.upper() for corpus in corpus_list),
            },
        )

    return _corpus_info


@pytest.fixture
def corpus_info_single(corpus_info: Callable[[list[str]], dict]) -> Callable[[str], dict]:
    """Return function returning `/corpora/info` response for a single corpus."""

    def _corpus_info_single(corpus: str) -> dict:
        """Return `/corpora/info` response for corpus (corpus-specific part)."""
        return corpus_info([corpus])["corpora"][corpus.lower()]

    return _corpus_info_single


class FakeAuthorizer:
    """A stand-in for an `Authorizer` that only lists protected corpora."""

    def __init__(self, protected: list[str]) -> None:
        """Initialize with the protected corpora."""
        self.protected = protected

    async def get_protected_corpora(self, auth_ctx: AuthContext) -> list[str]:  # ruff: ignore[unused-method-argument]
        """Return the protected corpora."""
        return self.protected


class TestInfo:
    """Tests for the `/info` endpoint."""

    @staticmethod
    def test_info_contains_version(get_json: Callable, corpora: list[str]) -> None:
        """Test that `/info` returns version info and canonical lowercase corpus ids."""
        data = get_json("/info")
        assert data["version"]
        assert set(data["corpora"]) == {corpus.lower() for corpus in corpora}
        assert all(corpus == corpus.lower() for corpus in data["protected_corpora"])

    @staticmethod
    def test_info_broken_registry_file(get_json: Callable, corpora: list[str], registry_with_broken_file: Path) -> None:
        """Test that `/info` leaves out a corpus whose registry file CQP cannot parse, instead of failing."""
        data = get_json("/info", params={"cache": "false"}, config={"CWB_REGISTRY": str(registry_with_broken_file)})
        assert set(data["corpora"]) == {corpus.lower() for corpus in corpora}

    @staticmethod
    @pytest.mark.parametrize("broken_file", [False, True], ids=["", "broken_file"])
    @pytest.mark.parametrize("exclude_undefined_corpora", [False, True], ids=["no_exclude", "exclude"])
    def test_info_undefined_corpus(
        get_json: Callable,
        corpora: list[str],
        registry_with_corpus_without_data: Path,
        exclude_undefined_corpora: bool,
        broken_file: bool,
    ) -> None:
        """Test that `/info` lists a corpus that is in the registry but undefined in CQP, unless excluded.

        With `exclude_undefined_corpora`, the corpus is left out, also if the registry has a file that CQP cannot
        parse.
        """
        if broken_file:
            write_broken_registry_file(registry_with_corpus_without_data)
        data = get_json(
            "/info",
            params={"cache": "false", "exclude_undefined_corpora": str(exclude_undefined_corpora).lower()},
            config={"CWB_REGISTRY": str(registry_with_corpus_without_data)},
        )
        expected = {corpus.lower() for corpus in corpora}
        if not exclude_undefined_corpora:
            expected.add(CORPUS_WITHOUT_DATA)
        assert set(data["corpora"]) == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("exclude_undefined_corpora", "expected_protected"),
        [(True, ["notinregistry", "testcorpus"]), (False, [CORPUS_WITHOUT_DATA, "notinregistry", "testcorpus"])],
        ids=["exclude", "no_exclude"],
    )
    def test_info_exclude_undefined_corpora_protected(
        client_factory: Callable[..., AbstractContextManager[TestClient]],
        registry_with_corpus_without_data: Path,
        exclude_undefined_corpora: bool,
        expected_protected: list[str],
    ) -> None:
        """Test that `/info` with `exclude_undefined_corpora` leaves out an undefined corpus from `protected_corpora`.

        A protected corpus that is not in the registry at all is listed in both cases.
        """
        with client_factory({"CWB_REGISTRY": str(registry_with_corpus_without_data)}) as client:
            cast(FastAPI, client.app).state.authorizer = FakeAuthorizer(
                ["testcorpus", CORPUS_WITHOUT_DATA, "notinregistry"]
            )
            data = get_response_json(
                client,
                "/info",
                params={"cache": "false", "exclude_undefined_corpora": str(exclude_undefined_corpora).lower()},
            )
        assert data["protected_corpora"] == expected_protected

    @staticmethod
    def test_info_unknown_parameter(get_json: Callable) -> None:
        """Test that `/info` rejects an unknown parameter."""
        data = get_json("/info", params={"no_such_param": "true"}, expected_status_code=422)
        assert data["code"] == "invalid_request"


class TestCorpusInfo:
    """Tests for the `/corpora/info` endpoint."""

    @staticmethod
    def _get_corpora_info_sum(data: dict, key: str) -> int:
        """Return sum of info item key for all corpora in data["corpora"]."""
        return sum(int(corpusdata["info"][key]) for corpusdata in data["corpora"].values())

    def test_corpus_info(self, corpus_info: Callable[[list[str]], dict], corpora: list[str]) -> None:
        """Test `/corpora/info` for all corpora."""
        data = corpus_info(corpora)
        assert len(data["corpora"]) == len(corpora)
        assert set(data["corpora"].keys()) == {corpus.lower() for corpus in corpora}
        assert data["total_size"] == self._get_corpora_info_sum(data, "size")
        assert data["total_sentences"] == self._get_corpora_info_sum(data, "sentences")

    @staticmethod
    def test_corpus_info_broken_registry_file(get_json: Callable, registry_with_broken_file: Path) -> None:
        """Test that a registry file that CQP cannot parse makes `/corpora/info` fail only for that corpus."""
        config = {"CWB_REGISTRY": str(registry_with_broken_file)}
        data = get_json("/corpora/info", params={"cache": "false", "corpora": "testcorpus"}, config=config)
        assert list(data["corpora"]) == ["testcorpus"]

        data = get_json(
            "/corpora/info",
            params={"cache": "false", "corpora": "testcorpus,brokencorpus"},
            config=config,
            expected_status_code=400,
        )
        assert data["code"] == "cqp_error"
        assert data["detail"] == "Corpus ``BROKENCORPUS'' is undefined"

    @staticmethod
    @pytest.mark.parametrize("undefined_corpus", [CORPUS_WITHOUT_DATA, "nosuchcorpus"])
    def test_corpus_info_undefined_corpus(
        get_json: Callable, registry_with_corpus_without_data: Path, undefined_corpus: str
    ) -> None:
        """Test that `/corpora/info` returns a CQP error for an undefined corpus.

        `CORPUS_WITHOUT_DATA` has a registry file but no data; `nosuchcorpus` has no registry file.
        """
        data = get_json(
            "/corpora/info",
            params={"cache": "false", "corpora": f"testcorpus,{undefined_corpus}"},
            config={"CWB_REGISTRY": str(registry_with_corpus_without_data)},
            expected_status_code=400,
        )
        assert data["code"] == "cqp_error"
        assert f"Corpus ``{undefined_corpus.upper()}'' is undefined" in data["detail"]

    @staticmethod
    @pytest.mark.parametrize("method", ["GET", "POST"])
    @pytest.mark.parametrize(
        ("expected_corpora", "expected_undefined"),
        [
            (["testcorpus"], [CORPUS_WITHOUT_DATA, "nosuchcorpus"]),
            (["testcorpus"], ["nosuchcorpus"]),
            (["testcorpus"], ["1abc", "a.b"]),
            (["testcorpus"], ["brokencorpus"]),
            ([], ["a.b", "nosuchcorpus"]),
            (["testcorpus", "testcorpus5"], []),
        ],
        ids=["no_data", "no_registry_file", "invalid_id", "broken_registry_file", "all_undefined", "all_defined"],
    )
    def test_corpus_info_report_undefined_corpora(
        client_factory: Callable[..., AbstractContextManager[TestClient]],
        registry_with_corpus_without_data: Path,
        method: str,
        expected_corpora: list[str],
        expected_undefined: list[str],
    ) -> None:
        """Test that `/corpora/info` with `report_undefined_corpora` lists the undefined corpora.

        `CORPUS_WITHOUT_DATA` and `brokencorpus` (whose registry file CQP cannot parse) have a registry file, so they
        are found only by selecting the corpora in CQP after getting the information fails. Ids that CQP cannot select
        are undefined.
        """
        write_broken_registry_file(registry_with_corpus_without_data)
        corpora = sorted(expected_corpora + expected_undefined)
        config = {"CWB_REGISTRY": str(registry_with_corpus_without_data)}
        with client_factory(config) as client:
            if method == "GET":
                response = client.get(
                    "/corpora/info",
                    params={"cache": "false", "corpora": ",".join(corpora), "report_undefined_corpora": "true"},
                )
            else:
                response = client.post(
                    "/corpora/info",
                    params={"cache": "false"},
                    json={"corpora": corpora, "report_undefined_corpora": True},
                )
            assert response.status_code == HTTPStatus.OK, response.text
            validate_response_contract(client, response, "/corpora/info", method=method)
        data = response.json()
        assert sorted(data["corpora"]) == expected_corpora
        assert data["undefined_corpora"] == expected_undefined
        assert data["total_size"] == sum(corpus["info"]["size"] for corpus in data["corpora"].values())
        assert data["total_sentences"] == sum(corpus["info"]["sentences"] for corpus in data["corpora"].values())

    @staticmethod
    @pytest.mark.skipif(os.geteuid() == 0, reason="root can read files without read permission")
    def test_corpus_info_report_undefined_corpora_unreadable_data(
        get_json: Callable, registry_with_corpus_without_data: Path, tmp_path: Path
    ) -> None:
        """Test that `/corpora/info` with `report_undefined_corpora` reports a corpus whose data file cannot be read.

        CWB prints its own error before CQP reports the corpus as undefined.
        """
        data_dir = tmp_path / "unreadablecorpus"
        copy_corpus(registry_with_corpus_without_data, "testcorpus", "unreadablecorpus", data_dir)
        (data_dir / "word.corpus").chmod(0)
        data = get_json(
            "/corpora/info",
            params={"cache": "false", "corpora": "testcorpus,unreadablecorpus", "report_undefined_corpora": "true"},
            config={"CWB_REGISTRY": str(registry_with_corpus_without_data)},
        )
        assert list(data["corpora"]) == ["testcorpus"]
        assert data["undefined_corpora"] == ["unreadablecorpus"]

    @staticmethod
    @pytest.mark.usefixtures("corpora")
    def test_corpus_info_report_undefined_corpora_not_requested(get_json: Callable) -> None:
        """Test that `/corpora/info` has no `undefined_corpora` by default."""
        data = get_json("/corpora/info", params={"cache": "false", "corpora": "testcorpus"})
        assert "undefined_corpora" not in data

    @staticmethod
    def test_corpus_info_unknown_parameter(get_json: Callable) -> None:
        """Test that `/corpora/info` rejects an unknown parameter."""
        data = get_json(
            "/corpora/info", params={"corpora": "testcorpus", "no_such_param": "true"}, expected_status_code=422
        )
        assert data["code"] == "invalid_request"

    @pytest.mark.parametrize(
        ("corpus", "attrs_p", "attrs_s", "attrs_a"),
        [
            (
                "testcorpus",
                ["word", "lemma"],
                ["text", "text_id", "paragraph", "paragraph_id", "sentence", "sentence_id"],
                [],
            ),
            (
                "testcorpus5",
                ["word", "lemma"],
                [
                    "text",
                    "paragraph",
                    "paragraph_id",
                    "sentence",
                    "sentence_id",
                    "sentence_a",
                    "span",
                    "span1",
                    "span2",
                    "span_n",
                    "span_n1",
                    "span_n2",
                ],
                [],
            ),
            (
                "testcorpus6",
                ["word", "baseform", "pos"],
                [
                    "corpus",
                    "corpus_id",
                    "document",
                    "text",
                    "paragraph",
                    "paragraph_id",
                    "paragraph_y",
                    "sentence",
                    "sentence_id",
                    "sentence_x",
                ],
                [],
            ),
        ],
    )
    @staticmethod
    def test_corpus_info_single_corpus(
        corpus: str,
        attrs_p: list[str],
        attrs_s: list[str],
        attrs_a: list[str],
        corpus_info_single: Callable[[str], dict],
    ) -> None:
        """Test `/corpora/info` for a single corpus."""
        data = corpus_info_single(corpus)
        attrs = data["attrs"]
        assert attrs
        assert attrs["positional"] == attrs_p
        assert attrs["structural"] == attrs_s
        assert attrs["alignment"] == attrs_a
        assert data["info"]["name"] == corpus.lower()


@pytest.fixture
def cached_client(
    cached_client_factory: Callable[..., AbstractContextManager[TestClient]], registry_with_corpus_without_data: Path
) -> Iterator[TestClient]:
    """Yield a TestClient with caching enabled, using the registry `registry_with_corpus_without_data`."""
    with cached_client_factory({"CWB_REGISTRY": str(registry_with_corpus_without_data)}) as client:
        yield client


class TestUndefinedCorporaCached:
    """Tests for checking undefined corpora in `/info` and `/corpora/info` with caching enabled."""

    @staticmethod
    def test_info_exclude_undefined_corpora_cached_separately(cached_client: TestClient) -> None:
        """Test that `/info` with and without `exclude_undefined_corpora` are cached separately."""
        excluding_params = {"exclude_undefined_corpora": "true", "debug": "true"}
        excluded = get_response_json(cached_client, "/info", params=excluding_params)
        included = get_response_json(cached_client, "/info", params={"debug": "true"})
        excluded_again = get_response_json(cached_client, "/info", params=excluding_params)
        included_again = get_response_json(cached_client, "/info", params={"debug": "true"})

        assert excluded["debug"].get("cache_saved")
        assert included["debug"].get("cache_saved")
        assert excluded_again["debug"].get("cache_read")
        assert included_again["debug"].get("cache_read")
        assert CORPUS_WITHOUT_DATA not in excluded["corpora"]
        assert CORPUS_WITHOUT_DATA not in excluded_again["corpora"]
        assert CORPUS_WITHOUT_DATA in included["corpora"]
        assert CORPUS_WITHOUT_DATA in included_again["corpora"]

    @staticmethod
    def test_corpus_info_report_undefined_corpora_cached(cached_client: TestClient) -> None:
        """Test that `undefined_corpora` is not added to the cached result of `/corpora/info` for the defined corpora.

        Both requests read the result for `testcorpus` from the cache.
        """
        params = {"corpora": f"testcorpus,{CORPUS_WITHOUT_DATA}", "report_undefined_corpora": "true", "debug": "true"}
        reported = get_response_json(cached_client, "/corpora/info", params=params)
        reported_again = get_response_json(cached_client, "/corpora/info", params=params)
        plain = get_response_json(cached_client, "/corpora/info", params={"corpora": "testcorpus", "debug": "true"})

        assert reported["undefined_corpora"] == reported_again["undefined_corpora"] == [CORPUS_WITHOUT_DATA]
        assert reported_again["debug"].get("cache_read")
        assert plain["debug"].get("cache_read")
        assert "undefined_corpora" not in plain
        assert list(plain["corpora"]) == list(reported["corpora"]) == ["testcorpus"]

    @staticmethod
    def test_added_corpus_defined(
        cached_client: TestClient, registry_with_corpus_without_data: Path, tmp_path: Path
    ) -> None:
        """Test that `/corpora/info` reports a corpus added to the registry as defined at once.

        The result of checking the registry files is not cached.
        """
        params = {"corpora": "testcorpus,newcorpus", "report_undefined_corpora": "true"}
        assert get_response_json(cached_client, "/corpora/info", params=params)["undefined_corpora"] == ["newcorpus"]

        copy_corpus(registry_with_corpus_without_data, "testcorpus", "newcorpus", tmp_path / "newcorpus")
        corpus_info = get_response_json(cached_client, "/corpora/info", params=params)
        assert corpus_info["undefined_corpora"] == []
        assert sorted(corpus_info["corpora"]) == ["newcorpus", "testcorpus"]

    @staticmethod
    def test_repaired_corpus_defined(
        cached_client: TestClient, registry_with_corpus_without_data: Path, tmp_path: Path
    ) -> None:
        """Test that a repaired corpus is defined in `/corpora/info` at once, but in `/info` after a cache refresh.

        `/corpora/info` selects the corpora in CQP, and uses the cached result, only if getting the information fails,
        whereas `/info` caches its whole result. The refresh invalidates the cached result only because `copy_corpus`
        rewrites the registry file of the repaired corpus.
        """
        corpus_info_params = {"corpora": f"testcorpus,{CORPUS_WITHOUT_DATA}", "report_undefined_corpora": "true"}
        info_params = {"exclude_undefined_corpora": "true"}

        def get_corpus_info_and_info() -> tuple[dict, dict]:
            return (
                get_response_json(cached_client, "/corpora/info", params=corpus_info_params),
                get_response_json(cached_client, "/info", params=info_params),
            )

        corpus_info, info = get_corpus_info_and_info()
        assert corpus_info["undefined_corpora"] == [CORPUS_WITHOUT_DATA]
        assert CORPUS_WITHOUT_DATA not in info["corpora"]

        # Repair the corpus by making it a copy of testcorpus
        copy_corpus(
            registry_with_corpus_without_data, "testcorpus", CORPUS_WITHOUT_DATA, tmp_path / CORPUS_WITHOUT_DATA
        )
        corpus_info, info = get_corpus_info_and_info()
        assert corpus_info["undefined_corpora"] == []
        assert sorted(corpus_info["corpora"]) == sorted(["testcorpus", CORPUS_WITHOUT_DATA])
        assert CORPUS_WITHOUT_DATA not in info["corpora"]

        response = cached_client.post("/admin/cache/refresh")
        assert response.status_code == HTTPStatus.OK, response.text
        assert response.json()["corpora_invalidated"] == 1
        _corpus_info, info = get_corpus_info_and_info()
        assert CORPUS_WITHOUT_DATA in info["corpora"]

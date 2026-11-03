"""Pytest tests for the Korp `/info` and `/corpora/info` endpoints."""

from collections.abc import Callable
from pathlib import Path

import pytest


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

"""Pytest tests for the Korp `/info` and `/corpus_info` endpoints."""

from collections.abc import Callable

import pytest


@pytest.fixture
def corpus_info(get_json: Callable, corpora: list[str]) -> Callable[[list[str]], dict]:  # noqa: ARG001
    """Return function returning `/corpus_info` response for list of corpora."""

    def _corpus_info(corpus_list: list[str]) -> dict:
        """Return `/corpus_info` response for the corpora in `corpus_list`."""
        return get_json(
            "/corpus_info",
            params={
                "cache": "false",
                "corpus": ",".join(corpus.upper() for corpus in corpus_list),
            },
        )

    return _corpus_info


@pytest.fixture
def corpus_info_single(corpus_info: Callable[[list[str]], dict]) -> Callable[[str], dict]:
    """Return function returning `/corpus_info` response for a single corpus."""

    def _corpus_info_single(corpus: str) -> dict:
        """Return `/corpus_info` response for corpus (corpus-specific part)."""
        return corpus_info([corpus])["corpora"][corpus.upper()]

    return _corpus_info_single


class TestInfo:
    """Tests for the `/info` endpoint."""

    @staticmethod
    def test_info_contains_version(get_json: Callable) -> None:
        """Test that `/info` response contains version info."""
        data = get_json("/info")
        assert data["version"]


class TestCorpusInfo:
    """Tests for the `/corpus_info` endpoint."""

    @staticmethod
    def _get_corpora_info_sum(data: dict, key: str) -> int:
        """Return sum of info item key for all corpora in data["corpora"]."""
        return sum(int(corpusdata["info"][key]) for corpusdata in data["corpora"].values())

    def test_corpus_info(self, corpus_info: Callable[[list[str]], dict], corpora: list[str]) -> None:
        """Test `/corpus_info` for all corpora."""
        data = corpus_info(corpora)
        assert len(data["corpora"]) == len(corpora)
        assert set(data["corpora"].keys()) == {corpus.upper() for corpus in corpora}
        assert data["total_size"] == self._get_corpora_info_sum(data, "Size")
        assert data["total_sentences"] == self._get_corpora_info_sum(data, "Sentences")

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
        """Test `/corpus_info` for a single corpus."""
        data = corpus_info_single(corpus)
        attrs = data["attrs"]
        assert attrs
        assert attrs["p"] == attrs_p
        assert attrs["s"] == attrs_s
        assert attrs["a"] == attrs_a

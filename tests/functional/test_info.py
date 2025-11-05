
"""
test_info.py

Pytest tests for the Korp /info and /corpus_info endpoints.
"""


import pytest

from tests.testutils import get_response_json


@pytest.fixture
def corpus_info(client, corpora):
    """Yield function returning /corpus_info response for list of corpora."""

    def _corpus_info(corpuslist):
        """Return /corpus_info response for the corpora in corpuslist."""
        data = get_response_json(
            client(), "/corpus_info",
            query_string={
                "cache": "false",
                "corpus": ",".join(corpus.upper() for corpus in corpuslist),
            })
        # print(data)
        return data

    yield _corpus_info


@pytest.fixture
def corpus_info_single(corpus_info):
    """Yield function returning /corpus_info response for a single corpus."""

    def _corpus_info_single(corpus):
        """Return /corpus_info response for corpus (corpus-specific part)."""
        return corpus_info([corpus])["corpora"][corpus.upper()]

    yield _corpus_info_single


class TestInfo:

    """Tests for the /info endpoint"""

    def test_info_contains_version(self, client):
        """Test that /info response contains version info."""
        data = get_response_json(client(), "/info")
        assert data["version"] and data["version"] != ""


class TestCorpusInfo:

    """Tests for the /corpus_info endpoint"""

    def _get_corpora_info_sum(self, data, key):
        """Return sum of info item key for all corpora in data["corpora"]."""
        return sum(int(corpusdata["info"][key])
                   for corpusdata in data["corpora"].values())

    def test_corpus_info(self, corpus_info, corpora):
        """Test /corpus_info for all corpora."""
        data = corpus_info(corpora)
        assert len(data["corpora"]) == len(corpora)
        assert (set(data["corpora"].keys())
                == set(corpus.upper() for corpus in corpora))
        assert data["total_size"] == self._get_corpora_info_sum(data, "Size")
        assert data["total_sentences"] == self._get_corpora_info_sum(
            data, "Sentences")

    @pytest.mark.parametrize(
        "corpus,attrs_p,attrs_s,attrs_a", [
            ("testcorpus",
             ["word", "lemma"],
             ["text", "text_id", "paragraph", "paragraph_id",
              "sentence", "sentence_id"],
             []),
        ])
    def test_corpus_info_single_corpus(self, corpus, attrs_p, attrs_s, attrs_a,
                                       corpus_info_single):
        """Test /corpus_info for a single corpus."""
        data = corpus_info_single(corpus)
        attrs = data["attrs"]
        assert attrs
        assert attrs["p"] == attrs_p
        assert attrs["s"] == attrs_s
        assert attrs["a"] == attrs_a

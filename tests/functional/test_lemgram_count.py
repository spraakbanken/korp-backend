
"""
test_lemgram_count.py

Pytest tests for the Korp /lemgram_count endpoint
"""


import pytest

from tests.testutils import get_response_json, make_liststr


@pytest.fixture
def lemgram_count(client, database_tables):
    """Yield function returning JSON response for /lemgram_count to corpora.

    The returned function takes as its parameters a lemgram, a corpus
    or a list of corpora, possible additional request parameters and
    Korp configuration parameters.
    It returns the JSON response for /lemgram_count to the specified
    corpora with the given parameters (and cache=false). It imports
    the lemgram_index database data for the given corpora.
    """

    def _lemgram_count(lemgram, corpora, params=None, config=None):
        query_params = {
            "corpus": make_liststr(corpora),
            "lemgram": lemgram,
            "cache": "false",
        }
        database_tables(corpora, "lemgram_index")
        query_params.update(params or {})
        return get_response_json(
            client(config or {}), "/lemgram_count", query_string=query_params)

    yield _lemgram_count


class TestLemgramCount:

    """Tests for /lemgram_count"""

    def test_lemgram_count_single_corpus(self, lemgram_count):
        """Test /lemgram_count on a single corpus and single lemgram."""
        lemgram = "test..nn.1"
        corpus = "testcorpus1"
        data = lemgram_count(lemgram, corpus)
        assert lemgram in data

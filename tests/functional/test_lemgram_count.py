"""Pytest tests for the Korp `/lemgram_count` endpoint."""

from collections.abc import Callable

import pytest

from tests.testutils import make_liststr


@pytest.fixture
def lemgram_count(get_json: Callable, database_tables: Callable) -> Callable:
    """Return function returning JSON response for `/lemgram_count` to corpora.

    The returned function takes as its parameters a lemgram, a corpus or a list of corpora, possible additional request
    parameters and Korp configuration parameters. It returns the JSON response for `/lemgram_count` to the specified
    corpora with the given parameters (and cache=false). It imports the lemgram_index database data for the given
    corpora.
    """

    def _lemgram_count(
        lemgram: str, corpora: list[str], params: dict | None = None, config: dict | None = None
    ) -> dict:
        query_params = {
            "corpus": make_liststr(corpora),
            "lemgram": lemgram,
            "cache": "false",
        }
        database_tables(corpora, "lemgram_index")
        query_params.update(params or {})
        return get_json("/lemgram_count", params=query_params, config=config)

    return _lemgram_count


class TestLemgramCount:
    """Tests for `/lemgram_count`."""

    @staticmethod
    def test_lemgram_count_single_corpus(lemgram_count: Callable) -> None:
        """Test `/lemgram_count` on a single corpus and single lemgram."""
        lemgram = "test..nn.1"
        corpus = "testcorpus1"
        data = lemgram_count(lemgram, corpus)
        assert lemgram in data

"""Pytest tests for the Korp `/lexeme_count` endpoint."""

from collections.abc import Callable

import pytest

from tests.testutils import make_liststr


@pytest.fixture
def lexeme_count(get_json: Callable, database_tables: Callable) -> Callable:
    """Return function returning JSON response for `/lexeme_count` to corpora.

    The returned function takes as its parameters a lexeme, a corpus or a list of corpora, possible additional request
    parameters and Korp configuration parameters. It returns the JSON response for `/lexeme_count` to the specified
    corpora with the given parameters (and cache=false). It imports the lexeme_counts database data for the given
    corpora.
    """

    def _lexeme_count(
        lexeme: str, corpora: list[str], params: dict | None = None, config: dict | None = None
    ) -> dict:
        query_params = {
            "corpus": make_liststr(corpora),
            "lexeme": lexeme,
            "cache": "false",
        }
        database_tables(corpora, "lexeme_counts")
        query_params.update(params or {})
        return get_json("/lexeme_count", params=query_params, config=config)

    return _lexeme_count


class TestLexemeCount:
    """Tests for `/lexeme_count`."""

    @staticmethod
    def test_lexeme_count_single_corpus(lexeme_count: Callable) -> None:
        """Test `/lexeme_count` on a single corpus and single lexeme."""
        lexeme = "test..nn.1"
        corpus = "testcorpus1"
        data = lexeme_count(lexeme, corpus)
        assert lexeme in data

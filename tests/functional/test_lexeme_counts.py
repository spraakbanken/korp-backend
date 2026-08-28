"""Pytest tests for the Korp `/lexeme_counts` endpoint."""

from collections.abc import Callable

import pytest

from tests.testutils import make_liststr


@pytest.fixture
def lexeme_counts(get_json: Callable, database_tables: Callable) -> Callable:
    """Return function returning JSON response for `/lexeme_counts` to corpora.

    The returned function takes as its parameters a lexeme, a corpus or a list of corpora, possible additional request
    parameters and Korp configuration parameters. It returns the JSON response for `/lexeme_counts` to the specified
    corpora with the given parameters (and cache=false). It imports the lexeme_counts database data for the given
    corpora.
    """

    def _lexeme_counts(
        lexeme: str, corpora: list[str], params: dict | None = None, config: dict | None = None
    ) -> dict:
        query_params = {
            "corpora": make_liststr(corpora),
            "lexeme": lexeme,
            "cache": "false",
        }
        database_tables(corpora, "lexeme_counts")
        query_params.update(params or {})
        return get_json("/lexeme_counts", params=query_params, config=config)

    return _lexeme_counts


class TestLexemeCounts:
    """Tests for `/lexeme_counts`."""

    @staticmethod
    def test_lexeme_counts_single_corpus(lexeme_counts: Callable) -> None:
        """Test `/lexeme_counts` on a single corpus and single lexeme."""
        lexeme = "test..nn.1"
        corpus = "testcorpus1"
        data = lexeme_counts(lexeme, corpus)
        assert lexeme in data["lexeme_counts"]

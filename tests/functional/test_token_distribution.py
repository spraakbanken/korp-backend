"""Pytest tests for the Korp `/token-distribution` endpoint."""

from collections.abc import Callable

import pytest

from tests.testutils import make_liststr


@pytest.fixture
def token_distribution(get_json: Callable, database_tables: Callable) -> Callable:
    """Return function returning JSON response for `/token-distribution` to given corpora.

    The returned function takes as its parameters a corpus (or corpora), possible additional query parameters and Korp
    configuration parameters. It returns the JSON response for `/token-distribution` to the corpora with the given
    parameters (and cache=false).
    """

    def _token_distribution(corpora: list[str] | str, params: dict | None = None, config: dict | None = None) -> dict:
        query_params = {
            "corpora": make_liststr(corpora),
            "cache": "false",
        }
        database_tables(corpora, "timedata")
        query_params.update(params or {})
        return get_json("/token-distribution", params=query_params, config=config)

    return _token_distribution


class TestTokenDistribution:
    """Tests for `/token-distribution`."""

    @pytest.mark.parametrize("granularity", ["year", "month", "day", "hour", "minute", "second"])
    @staticmethod
    def test_token_distribution_granularity(granularity: str, token_distribution: Callable[..., dict]) -> None:
        """Test `/token-distribution` with granularity on testcorpus3 and testcorpus4."""
        corpora = ["testcorpus3", "testcorpus4"]
        data = token_distribution(corpora, {"granularity": granularity})
        assert "combined" in data
        assert "corpora" in data
        for corpus in corpora:
            assert corpus.upper() in data["corpora"]

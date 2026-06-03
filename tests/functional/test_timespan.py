"""Pytest tests for the Korp `/timespan` endpoint."""

from collections.abc import Callable

import pytest

from tests.testutils import make_liststr


@pytest.fixture
def timespan(get_json: Callable, database_tables: Callable) -> Callable:
    """Return function returning JSON response for `/timespan` to given corpora.

    The returned function takes as its parameters a corpus (or corpora), possible additional query parameters and Korp
    configuration parameters. It returns the JSON response for `/timespan` to the corpora with the given parameters (and
    cache=false).
    """

    def _timespan(corpora: list[str] | str, params: dict | None = None, config: dict | None = None) -> dict:
        query_params = {
            "corpus": make_liststr(corpora),
            "cache": "false",
        }
        database_tables(corpora, "timedata")
        query_params.update(params or {})
        return get_json("/timespan", params=query_params, config=config)

    return _timespan


class TestTimespan:
    """Tests for `/timespan`."""

    @pytest.mark.parametrize("granularity", ["year", "month", "day", "hour", "minute", "second"])
    @staticmethod
    def test_timespan_granularity(granularity: str, timespan: Callable[..., dict]) -> None:
        """Test `/timespan` with granularity on testcorpus3 and testcorpus4."""
        corpora = ["testcorpus3", "testcorpus4"]
        data = timespan(corpora, {"granularity": granularity})
        assert "combined" in data
        assert "corpora" in data
        for corpus in corpora:
            assert corpus.upper() in data["corpora"]

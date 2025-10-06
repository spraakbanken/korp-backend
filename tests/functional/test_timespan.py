
"""
test_timespan.py

Pytest tests for the Korp /timespan endpoint
"""


import pytest

from tests.testutils import get_response_json, make_liststr


@pytest.fixture
def timespan(client, database_tables):
    """Yield function returning JSON response for /timespan to given corpora.

    The returned function takes as its parameters a corpus (or
    corpora), possible additional query parameters and Korp
    configuration parameters. It returns the JSON response for
    /timespan to the corpora with the given parameters (and
    cache=false).
    """

    def _timespan(corpora, params=None, config=None):
        query_params = {
            "corpus": make_liststr(corpora),
            "cache": "false",
        }
        database_tables(corpora, "timedata")
        query_params.update(params or {})
        return get_response_json(
            client(config or {}), "/timespan", query_string=query_params)

    yield _timespan


class TestTimespan:

    """Tests for /timespan"""

    @pytest.mark.parametrize("granularity", ["y", "m", "d", "h", "n", "s"])
    def test_timespan_granularity(self, granularity, timespan):
        """Test /timespan with granularity on testcorpus3 and testcorpus4."""
        corpora = ["testcorpus3", "testcorpus4"]
        data = timespan(corpora, {"granularity": granularity})
        assert "combined" in data
        assert "corpora" in data
        for corpus in corpora:
            assert corpus.upper() in data["corpora"]

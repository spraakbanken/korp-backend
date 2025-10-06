
"""
test_query.py

Pytest tests for the Korp /query endpoint
"""


import pytest

from tests.testutils import get_response_json


@pytest.fixture
def query_testcorpus(client):
    """Yield function returning JSON response for /query to testcorpus.

    The returned function takes as its parameters the CQP query,
    possible additional query parameters and Korp configuration
    parameters. It returns the JSON response for /query to corpus
    "testcorpus" with the given parameters (and cache=false).
    """

    def _query_testcorpus(cqp, params=None, config=None):
        query = {
            "corpus": "testcorpus",
            "cqp": cqp,
            "cache": "false",
        }
        query.update(params or {})
        return get_response_json(
            client(config or {}), "/query", query_string=query)

    yield _query_testcorpus


@pytest.fixture
def query_testcorpus_kwic_rows(query_testcorpus):
    """Yield a function to test the effect of MAX_KWIC_ROWS.

    The returned function takes as its parameters the value for
    MAX_KWIC_ROWS and the number of rows to request. It returns the
    JSON response for /query to corpus "testcorpus" with CQP query
    "[]" (any word) from the beginning of the corpus (start=0).
    """

    def _query_testcorpus_kwic_rows(max_rows, request_rows):
        return query_testcorpus(
            "[]",
            {
                "start": "0",
                "end": str(request_rows - 1),
            },
            {"MAX_KWIC_ROWS": max_rows}
        )

    yield _query_testcorpus_kwic_rows


class TestQuery:

    """Tests for /query"""

    def test_query_single_corpus(self, query_testcorpus):
        """Test a simple query on a single corpus."""
        data = query_testcorpus("[lemma=\"this\"]")
        kwic = data["kwic"]
        assert len(kwic) == data["hits"]
        # print(data)
        # assert 0

    def test_query_max_kwic_rows(self, query_testcorpus_kwic_rows):
        """Test a query requesting MAX_KWIC_ROWS results."""
        num = 1
        data = query_testcorpus_kwic_rows(num, num)
        assert len(data["kwic"]) == num

    def test_query_max_kwic_exceeded(self, query_testcorpus_kwic_rows):
        """Test a query requesting MAX_KWIC_ROWS + 1 results."""
        num = 1
        data = query_testcorpus_kwic_rows(num, num + 1)
        errmsg = f"At most {num} KWIC rows can be returned per call."
        assert "ERROR" in data and errmsg in data["ERROR"]["value"]

    def test_query_max_kwic_unlimited(self, query_testcorpus_kwic_rows):
        """Test a query with MAX_KWIC_ROWS = 0."""
        # testcorpus does not contain 1,000,000 tokens, so the
        # following should return all hits. MAX_KWIC_ROWS is tested
        # before returning the data.
        data = query_testcorpus_kwic_rows(0, 1000000)
        assert len(data["kwic"]) == data["hits"]

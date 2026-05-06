"""Pytest tests for the Korp `/query` endpoint."""

from collections.abc import Callable

import pytest


@pytest.fixture
def query_testcorpus(get_json: Callable, corpora: list[str]) -> Callable[..., dict]:  # noqa: ARG001
    """Return function returning JSON response for `/query` to testcorpus.

    The returned function takes as its parameters the CQP query, possible additional query parameters and Korp
    configuration parameters. It returns the JSON response for `/query` to corpus "testcorpus" with the given parameters
    (and cache=false).
    """

    def _query_testcorpus(cqp: str, params: dict | None = None, config: dict | None = None) -> dict:
        query = {
            "corpus": "testcorpus",
            "cqp": cqp,
            "cache": "false",
        }
        query.update(params or {})
        return get_json("/query", params=query, config=config)

    return _query_testcorpus


@pytest.fixture
def query_testcorpus_kwic_rows(
    query_testcorpus: Callable[..., dict],
) -> Callable[[int, int], dict]:
    """Return a function to test the effect of `MAX_KWIC_ROWS`.

    The returned function takes as its parameters the value for `MAX_KWIC_ROWS` and the number of rows to request. It
    returns the JSON response for `/query` to corpus "testcorpus" with CQP query "[]" (any word) from the beginning of
    the corpus (start=0).
    """

    def _query_testcorpus_kwic_rows(max_rows: int, request_rows: int) -> dict:
        return query_testcorpus(
            "[]",
            {
                "start": "0",
                "end": str(request_rows - 1),
            },
            {"MAX_KWIC_ROWS": max_rows},
        )

    return _query_testcorpus_kwic_rows


class TestQuery:
    """Tests for `/query`."""

    @staticmethod
    def test_query_single_corpus(query_testcorpus: Callable[..., dict]) -> None:
        """Test a simple query on a single corpus."""
        data = query_testcorpus('[lemma="this"]')
        kwic = data["kwic"]
        assert len(kwic) == data["hits"]

    @staticmethod
    def test_query_max_kwic_rows(query_testcorpus_kwic_rows: Callable[[int, int], dict]) -> None:
        """Test a query requesting `MAX_KWIC_ROWS` results."""
        num = 1
        data = query_testcorpus_kwic_rows(num, num)
        assert len(data["kwic"]) == num

    @staticmethod
    def test_query_max_kwic_exceeded(query_testcorpus_kwic_rows: Callable[[int, int], dict]) -> None:
        """Test a query requesting `MAX_KWIC_ROWS` + 1 results."""
        num = 1
        data = query_testcorpus_kwic_rows(num, num + 1)
        errmsg = f"At most {num} KWIC rows can be returned per call."
        assert "ERROR" in data
        assert errmsg in data["ERROR"]["value"]

    @staticmethod
    def test_query_max_kwic_unlimited(query_testcorpus_kwic_rows: Callable[[int, int], dict]) -> None:
        """Test a query with `MAX_KWIC_ROWS` = 0."""
        # testcorpus does not contain 1,000,000 tokens, so the following should return all hits. MAX_KWIC_ROWS is tested
        # before returning the data.
        data = query_testcorpus_kwic_rows(0, 1000000)
        assert len(data["kwic"]) == data["hits"]

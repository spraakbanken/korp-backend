"""Pytest tests for the Korp `/concordance` endpoint."""

from collections.abc import Callable

import pytest


@pytest.fixture
def concordance_testcorpus(get_json: Callable, corpora: list[str]) -> Callable[..., dict]:  # noqa: ARG001
    """Return function returning JSON response for `/concordance` to testcorpus.

    The returned function takes as its parameters the CQP query, possible additional query parameters and Korp
    configuration parameters. It returns the JSON response for `/concordance` to corpus "testcorpus" with the given
    parameters (and cache=false).
    """

    def _concordance_testcorpus(cqp: str, params: dict | None = None, config: dict | None = None) -> dict:
        query = {
            "corpora": "testcorpus",
            "cqp": cqp,
            "cache": "false",
        }
        query.update(params or {})
        return get_json("/concordance", params=query, config=config)

    return _concordance_testcorpus


@pytest.fixture
def concordance_sample_testcorpus(get_json: Callable, corpora: list[str]) -> Callable[..., dict]:  # noqa: ARG001
    """Return function returning JSON response for `/concordance/sample` to testcorpus."""

    def _concordance_sample_testcorpus(cqp: str, params: dict | None = None, config: dict | None = None) -> dict:
        query = {
            "corpora": "testcorpus",
            "cqp": cqp,
            "cache": "false",
        }
        query.update(params or {})
        return get_json("/concordance/sample", params=query, config=config)

    return _concordance_sample_testcorpus


@pytest.fixture
def concordance_testcorpus_kwic_rows(
    concordance_testcorpus: Callable[..., dict],
) -> Callable[[int, int], dict]:
    """Return a function to test the effect of `MAX_KWIC_ROWS`.

    The returned function takes as its parameters the value for `MAX_KWIC_ROWS` and the number of rows to request. It
    returns the JSON response for `/concordance` to corpus "testcorpus" with CQP query "[]" (any word) from the
    beginning of
    the corpus (offset=0).
    """

    def _concordance_testcorpus_kwic_rows(max_rows: int, request_rows: int) -> dict:
        return concordance_testcorpus(
            "[]",
            {
                "offset": "0",
                "limit": str(request_rows),
            },
            {"MAX_KWIC_ROWS": max_rows},
        )

    return _concordance_testcorpus_kwic_rows


class TestConcordance:
    """Tests for `/concordance`."""

    @staticmethod
    def test_concordance_single_corpus(concordance_testcorpus: Callable[..., dict]) -> None:
        """Test a simple concordance search on a single corpus."""
        data = concordance_testcorpus('[lemma="this"]')
        kwic = data["kwic"]
        assert len(kwic) == data["hits"]

    @staticmethod
    def test_concordance_max_kwic_rows(concordance_testcorpus_kwic_rows: Callable[[int, int], dict]) -> None:
        """Test a concordance search requesting `MAX_KWIC_ROWS` results."""
        num = 1
        data = concordance_testcorpus_kwic_rows(num, num)
        assert len(data["kwic"]) == num

    @staticmethod
    def test_concordance_max_kwic_exceeded(concordance_testcorpus_kwic_rows: Callable[[int, int], dict]) -> None:
        """Test a concordance search requesting `MAX_KWIC_ROWS` + 1 results."""
        num = 1
        data = concordance_testcorpus_kwic_rows(num, num + 1)
        errmsg = f"At most {num} KWIC rows can be returned per call."
        assert "error" in data
        assert errmsg in data["error"]["value"]

    @staticmethod
    def test_concordance_max_kwic_unlimited(concordance_testcorpus_kwic_rows: Callable[[int, int], dict]) -> None:
        """Test a concordance search with `MAX_KWIC_ROWS` = 0."""
        # testcorpus does not contain 1,000,000 tokens, so the following should return all hits. MAX_KWIC_ROWS is tested
        # before returning the data.
        data = concordance_testcorpus_kwic_rows(0, 1000000)
        assert len(data["kwic"]) == data["hits"]

    @staticmethod
    def test_concordance_invalid_cqp_surfaces_cqp_error(concordance_testcorpus: Callable[..., dict]) -> None:
        """Test that invalid CQP in parallel concordance search exposes the underlying CQP error."""
        data = concordance_testcorpus("unquoted")

        assert data["error"]["type"] == "CQPError"
        assert "Corpus ``unquoted'' is undefined" in data["error"]["value"]


class TestConcordanceSample:
    """Tests for `/concordance/sample`."""

    @staticmethod
    def test_concordance_sample_no_hit_returns_empty_kwic(concordance_sample_testcorpus: Callable[..., dict]) -> None:
        """Test that a sample concordance search with no matches returns an empty KWIC list."""
        data = concordance_sample_testcorpus('[word="__definitely_not_in_testcorpus__"]')

        assert data["kwic"] == []
        assert "hits" not in data
        assert "corpus_hits" not in data
        assert "pagination_state" not in data

    @staticmethod
    def test_concordance_sample_hit_omits_hit_counts(concordance_sample_testcorpus: Callable[..., dict]) -> None:
        """Test that a sample hit does not expose capped per-corpus hit counts."""
        data = concordance_sample_testcorpus("[]")

        assert len(data["kwic"]) == 1
        assert data["corpus_order"] == ["TESTCORPUS"]
        assert "hits" not in data
        assert "corpus_hits" not in data
        assert "pagination_state" not in data

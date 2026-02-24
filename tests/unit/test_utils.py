"""Unit tests for utility functions in korp.utils."""

import pytest
from pydantic import BaseModel

from korp import utils
from korp.api import params


class M(BaseModel):
    """Model for testing CorpusParam parsing."""

    corpora: params.CorpusParam


class TestCorpusParam:
    """Tests for the CorpusParam type."""

    @staticmethod
    def test_corpus_param_empty() -> None:
        """Assert that empty corpus parameter results in empty list."""
        m = M.model_validate({"corpora": []})
        assert m.corpora == []

    @staticmethod
    def test_corpus_param_empty_string() -> None:
        """Assert that empty string corpus parameter results in empty list."""
        m = M.model_validate({"corpora": ""})
        assert m.corpora == []

    @staticmethod
    def test_corpus_param_list() -> None:
        """Assert that list of corpora is parsed correctly."""
        m = M.model_validate({"corpora": ["A", "B"]})
        assert m.corpora == ["A", "B"]

    @staticmethod
    def test_corpus_param_string() -> None:
        """Assert that comma-separated string of corpora is parsed correctly."""
        m = M.model_validate({"corpora": utils.QUERY_DELIM.join(["A", "B"])})
        assert m.corpora == ["A", "B"]

    @staticmethod
    def test_corpus_param_string_upper() -> None:
        """Assert that corpus names are converted to uppercase."""
        m = M.model_validate({"corpora": utils.QUERY_DELIM.join(["a", "b"])})
        assert m.corpora == ["A", "B"]

    @staticmethod
    def test_corpus_param_sort() -> None:
        """Assert that corpus names are sorted."""
        m = M.model_validate({"corpora": ["B", "A"]})
        assert m.corpora == ["A", "B"]

    @staticmethod
    def test_corpus_param_unique() -> None:
        """Assert that duplicate corpus names are removed."""
        m = M.model_validate({"corpora": ["A", "B", "A", "B"]})
        assert m.corpora == ["A", "B"]

    @staticmethod
    def test_corpus_param_split_comma() -> None:
        """Assert that we handle both comma-separated corpora and multiple corpus parameters."""
        m = M.model_validate({"corpora": ["A,c", "b"]})
        assert m.corpora == ["A", "B", "C"]


@pytest.fixture(autouse=True)
def fixed_querylock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use deterministic query lock values in tests."""

    def _fake_randrange(_start: int, _stop: int) -> int:
        return 123456789

    monkeypatch.setattr(utils.random, "randrange", _fake_randrange)


class TestParseCQP:
    """Tests for the `parse_cqp` function."""

    @staticmethod
    def test_parse_cqp_extracts_bracket_tokens() -> None:
        """Return bracketed tokens and no rest for a simple sequence."""
        tokens, rest = utils.parse_cqp('[word="a"] [word="b"]')

        assert tokens == ['[word="a"]', '[word="b"]']
        assert rest is False

    @staticmethod
    def test_parse_cqp_extracts_quoted_and_repeated_tokens() -> None:
        """Handle quoted tokens and preserve repetition suffixes."""
        tokens, rest = utils.parse_cqp('"it""s" [word="x"]{2}')

        assert tokens == ['"it""s"', '[word="x"]{2}']
        assert rest is False

    @staticmethod
    def test_parse_cqp_detects_non_token_content() -> None:
        """Mark parsing as partial when non-whitespace content appears between tokens."""
        tokens, rest = utils.parse_cqp('[word="a"] | [word="b"]')

        assert tokens == ['[word="a"]', '[word="b"]']
        assert rest is True

    @staticmethod
    def test_parse_cqp_rejects_zero_width_assertion() -> None:
        """Return partial failure for zero-width assertions."""
        tokens, rest = utils.parse_cqp('[:word="x":] [word="y"]')

        assert tokens == []
        assert rest is True


class TestQueryOptimize:
    """Tests for the `optimize_query` function."""

    @staticmethod
    def test_optimize_query_returns_fallback_for_single_token() -> None:
        """Return NOT_NEEDED and fallback query when optimization is unnecessary."""
        retcode, query = utils.optimize_query('[word="a"]', {"within": "sentence"})

        assert retcode == utils.QueryOptimizeResult.NOT_NEEDED
        assert query == [
            "set QueryLock 123456789;",
            '[word="a"] within sentence;',
            "unlock 123456789;",
        ]

    @staticmethod
    def test_optimize_query_returns_fallback_when_within_is_missing() -> None:
        """Return NOT_POSSIBLE when required `within` context is missing."""
        retcode, query = utils.optimize_query('[word="a"] [word="b"]', {})

        assert retcode == utils.QueryOptimizeResult.NOT_POSSIBLE
        assert query == [
            "set QueryLock 123456789;",
            '[word="a"] [word="b"];',
            "unlock 123456789;",
        ]

    @staticmethod
    def test_optimize_query_builds_ordered_mu_without_expand() -> None:
        """Build a direct MU command when expansion and re-match are disabled."""
        retcode, query = utils.optimize_query(
            '[word="a"] [word="b"]',
            {"within": "sentence"},
            find_match=False,
            expand=False,
        )

        assert retcode == utils.QueryOptimizeResult.SUCCESS
        assert query == ['MU (meet [word="a"] [word="b"] 1 1);']

    @staticmethod
    def test_optimize_query_builds_wildcard_distance_range() -> None:
        """Convert wildcard repetitions to MU distance ranges."""
        retcode, query = utils.optimize_query(
            '[word="a"] []{2,4} [word="b"]',
            {"within": "sentence"},
            find_match=False,
            expand=False,
        )

        assert retcode == utils.QueryOptimizeResult.SUCCESS
        assert query == ['MU (meet [word="a"] [word="b"] 3 5);']

    @staticmethod
    def test_optimize_query_uses_within_distance_for_unbounded_wildcard() -> None:
        """Use `within` context as upper distance bound for unbounded wildcard ranges."""
        retcode, query = utils.optimize_query(
            '[word="a"] []{2,} [word="b"]',
            {"within": "sentence"},
            find_match=False,
            expand=False,
        )

        assert retcode == utils.QueryOptimizeResult.SUCCESS
        assert query == ['MU (meet [word="a"] [word="b"] sentence);']

    @staticmethod
    def test_optimize_query_uses_full_expand_when_leading_wildcard_exists() -> None:
        """Use full expand direction when leading wildcards were stripped."""
        retcode, query = utils.optimize_query('[] [word="a"] [word="b"]', {"within": "sentence"})

        assert retcode == utils.QueryOptimizeResult.SUCCESS
        assert query == [
            'MU (meet [word="a"] [word="b"] 1 1) expand to sentence;',
            "Last;",
            "set QueryLock 123456789;",
            '[] [word="a"] [word="b"] within sentence;',
            "unlock 123456789;",
        ]

    @staticmethod
    def test_optimize_query_rejects_wildcards_in_free_search() -> None:
        """Raise error when free-order search contains wildcard tokens."""
        with pytest.raises(utils.CQPError, match="Wildcards not allowed in free order query"):
            utils.optimize_query('[] [word="a"] [word="b"]', {"within": "sentence"}, free_search=True)

    @staticmethod
    def test_optimize_query_builds_free_search_mu_query() -> None:
        """Build free-order MU query using within-based constraints."""
        retcode, query = utils.optimize_query('[word="a"] [word="b"]', {"within": "sentence"}, free_search=True)

        assert retcode == utils.QueryOptimizeResult.SUCCESS
        assert query == ['MU (meet [word="a"] [word="b"] sentence) expand to sentence;']

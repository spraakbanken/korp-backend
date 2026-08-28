"""Unit tests for parameter parsing in korp.api.params."""

from pydantic import BaseModel

from korp import utils
from korp.api import params


class M(BaseModel):
    """Model for testing CorporaParam parsing."""

    corpora: params.CorporaParam


class TestCorporaParam:
    """Tests for the CorporaParam type."""

    @staticmethod
    def test_corpora_param_empty() -> None:
        """Assert that an empty corpora parameter results in an empty list."""
        m = M.model_validate({"corpora": []})
        assert m.corpora == []

    @staticmethod
    def test_corpora_param_empty_string() -> None:
        """Assert that an empty string corpora parameter results in an empty list."""
        m = M.model_validate({"corpora": ""})
        assert m.corpora == []

    @staticmethod
    def test_corpora_param_list() -> None:
        """Assert that a list of corpora is parsed correctly."""
        m = M.model_validate({"corpora": ["A", "B"]})
        assert m.corpora == ["A", "B"]

    @staticmethod
    def test_corpora_param_string() -> None:
        """Assert that comma-separated string of corpora is parsed correctly."""
        m = M.model_validate({"corpora": utils.QUERY_DELIM.join(["A", "B"])})
        assert m.corpora == ["A", "B"]

    @staticmethod
    def test_corpora_param_string_upper() -> None:
        """Assert that corpus names are converted to uppercase."""
        m = M.model_validate({"corpora": utils.QUERY_DELIM.join(["a", "b"])})
        assert m.corpora == ["A", "B"]

    @staticmethod
    def test_corpora_param_sort() -> None:
        """Assert that corpus names are sorted."""
        m = M.model_validate({"corpora": ["B", "A"]})
        assert m.corpora == ["A", "B"]

    @staticmethod
    def test_corpora_param_unique() -> None:
        """Assert that duplicate corpus names are removed."""
        m = M.model_validate({"corpora": ["A", "B", "A", "B"]})
        assert m.corpora == ["A", "B"]

    @staticmethod
    def test_corpora_param_split_comma() -> None:
        """Assert that we handle both comma-separated corpora and multiple corpus parameters."""
        m = M.model_validate({"corpora": ["A,c", "b"]})
        assert m.corpora == ["A", "B", "C"]

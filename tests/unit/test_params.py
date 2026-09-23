"""Unit tests for parameter parsing in korp.api.params."""

import pytest
from pydantic import BaseModel, ValidationError

from korp import utils
from korp.api import params


class M(BaseModel):
    """Model for testing CorporaParam parsing."""

    corpora: params.CorporaParam


class TestCorporaParam:
    """Tests for the CorporaParam type."""

    @staticmethod
    def test_corpora_param_empty() -> None:
        """Assert that an empty corpora parameter raises a validation error."""
        with pytest.raises(ValidationError):
            M.model_validate({"corpora": []})

    @staticmethod
    def test_corpora_param_empty_string() -> None:
        """Assert that an empty string corpora parameter raises a validation error."""
        with pytest.raises(ValidationError):
            M.model_validate({"corpora": ""})

    @staticmethod
    def test_corpora_param_list() -> None:
        """Assert that a list of corpora is parsed correctly."""
        m = M.model_validate({"corpora": ["A", "B"]})
        assert m.corpora == ["a", "b"]

    @staticmethod
    def test_corpora_param_string() -> None:
        """Assert that comma-separated string of corpora is parsed correctly."""
        m = M.model_validate({"corpora": utils.QUERY_DELIM.join(["A", "B"])})
        assert m.corpora == ["a", "b"]

    @staticmethod
    def test_corpora_param_string_lower() -> None:
        """Assert that corpus names are converted to lowercase."""
        m = M.model_validate({"corpora": utils.QUERY_DELIM.join(["A", "b"])})
        assert m.corpora == ["a", "b"]

    @staticmethod
    def test_corpora_param_sort() -> None:
        """Assert that corpus names are sorted."""
        m = M.model_validate({"corpora": ["B", "A"]})
        assert m.corpora == ["a", "b"]

    @staticmethod
    def test_corpora_param_unique() -> None:
        """Assert that duplicate corpus names are removed."""
        m = M.model_validate({"corpora": ["A", "B", "A", "B"]})
        assert m.corpora == ["a", "b"]

    @staticmethod
    def test_corpora_param_split_comma() -> None:
        """Assert that we handle both comma-separated corpora and multiple corpus parameters."""
        m = M.model_validate({"corpora": ["A,c", "b"]})
        assert m.corpora == ["a", "b", "c"]

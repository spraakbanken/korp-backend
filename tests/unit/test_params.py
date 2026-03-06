"""Unit tests for parameter parsing in korp.api.params."""

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

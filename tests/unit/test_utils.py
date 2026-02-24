"""Unit tests for utility functions in korp.utils."""

from pydantic import BaseModel

from korp import utils
from korp.api import params


class M(BaseModel):
    """Model for testing CorpusParam parsing."""
    corpora: params.CorpusParam


class TestCorpusParam:
    """Tests for the CorpusParam type."""

    def test_corpus_param_empty(self):
        m = M.model_validate({"corpora": []})
        assert m.corpora == []

    def test_corpus_param_empty_string(self):
        m = M.model_validate({"corpora": ""})
        assert m.corpora == []

    def test_corpus_param_list(self):
        m = M.model_validate({"corpora": ["A", "B"]})
        assert m.corpora == ["A", "B"]

    def test_corpus_param_string(self):
        m = M.model_validate({"corpora": utils.QUERY_DELIM.join(["A", "B"])})
        assert m.corpora == ["A", "B"]

    def test_corpus_param_string_upper(self):
        m = M.model_validate({"corpora": utils.QUERY_DELIM.join(["a", "b"])})
        assert m.corpora == ["A", "B"]

    def test_corpus_param_sort(self):
        m = M.model_validate({"corpora": ["B", "A"]})
        assert m.corpora == ["A", "B"]

    def test_corpus_param_unique(self):
        m = M.model_validate({"corpora": ["A", "B", "A", "B"]})
        assert m.corpora == ["A", "B"]

    def test_corpus_param_split_comma(self):
        m = M.model_validate({"corpora": ["A,c", "b"]})
        assert m.corpora == ["A", "B", "C"]


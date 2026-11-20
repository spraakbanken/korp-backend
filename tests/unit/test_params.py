"""Unit tests for parameter parsing in korp.api.params."""

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from korp import utils
from korp.api import params
from korp.api.routers import attribute_values, frequencies, query


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

    @staticmethod
    def test_corpora_param_aligned() -> None:
        """Accept aligned corpus IDs with underscores, hyphens and digits."""
        m = M.model_validate({"corpora": [" _A-1|B_2 "]})
        assert m.corpora == ["_a-1|b_2"]

    @staticmethod
    @pytest.mark.parametrize("corpus", ["corpusé", "ärtal", "corpus٣", "corpus.v1", "1corpus", "-corpus", "a|2b"])
    def test_corpora_param_invalid_id(corpus: str) -> None:
        """Reject corpus IDs outside the CQP identifier syntax."""
        with pytest.raises(ValidationError):
            M.model_validate({"corpora": [corpus]})


def test_corpus_context() -> None:
    """Apply the same corpus ID rules to per-corpus contexts."""
    adapter = TypeAdapter(params.CorpusContext)
    assert adapter.validate_python("_A-1|B_2:10 word") == "_A-1|B_2:10 word"
    with pytest.raises(ValidationError):
        adapter.validate_python("_A-1|B.2:10 word")


@pytest.mark.parametrize(
    ("parameter", "valid", "invalid"),
    [
        (params.Attribute, "_lemma-1", "lemma.v1"),
        (params.Attribute, "_lemma-1", "1lemma"),
        (params.Attribute, "_lemma-1", "-lemma"),
        (attribute_values.AttrParam, ["text_author>text_title"], ["text_author>text.title"]),
        (attribute_values.AttrParam, ["text_author>text_title"], ["text.author>text_title"]),
        (frequencies.MaxValuesPerSetParam, ["sense-1:3", "lemma"], ["sense.v1:3"]),
        (params.SplitParam, ["sense-1"], ["sense.v1"]),
        (params.DefaultContextParam, "1 text_section", "1 text.section"),
        (params.CorpusContext, "test:1 text_section", "test:1 text.section"),
        (params.DefaultWithinParam, "2 text_section", "2 text.section"),
        (params.DefaultWithinParam, "2 `text_section`", "2 `text.section`"),
        (params.WithinParam, ["test:2 text_section", "other:"], ["test:2 text.section"]),
        (query.OptimizeWithinParam, "2", "text.section"),
    ],
)
def test_attribute_identifiers(parameter: type, valid: object, invalid: object) -> None:
    """Apply the CQP identifier rule to attributes in plain and compound parameters."""
    adapter = TypeAdapter(parameter)
    assert adapter.validate_python(valid) == valid
    with pytest.raises(ValidationError):
        adapter.validate_python(invalid)

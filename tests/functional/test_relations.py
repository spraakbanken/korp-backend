
"""Pytest tests for the Korp /relations endpoint."""

import pytest

from tests.testutils import make_liststr


@pytest.fixture
def relations_testcorpus(get_json, database_tables):
    """Return function returning JSON response for /relations to testcorpus.

    The returned function takes as its parameters a word, corpus (or
    corpora), possible additional query parameters and Korp
    configuration parameters. It returns the JSON response for
    /relations with the given parameters (and cache=false).
    """

    def _relations_testcorpus(word, corpora, params=None, config=None):
        query_params = {
            "corpus": make_liststr(corpora),
            "word": word,
            "cache": "false",
            "measures": "freq,mi"
        }
        database_tables(corpora, "relations")
        query_params.update(params or {})
        return get_json("/relations", params=query_params, config=config)

    return _relations_testcorpus


class TestRelations:
    """Tests for /relations."""

    @pytest.mark.parametrize("word", ["är"])
    @pytest.mark.parametrize("corpora", ["testcorpus2",
                                         ["testcorpus2", "testcorpus2b"]])
    def test_relations_simple(self, word, corpora, relations_testcorpus):
        """Test /relations with the given word and corpora."""
        data = relations_testcorpus(word, corpora)
        assert "relations" in data
        for rel in data["relations"]:
            assert rel["head"] == word or rel["dep"] == word

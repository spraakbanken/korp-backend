"""Pytest tests for the Korp `/dependency-relations` endpoint."""

from collections.abc import Callable

import pytest

from tests.testutils import make_liststr


@pytest.fixture
def dependency_relations_testcorpus(get_json: Callable, database_tables: Callable) -> Callable:
    """Return function returning JSON response for `/dependency-relations` to testcorpus.

    The returned function takes as its parameters a word, corpus (or corpora), possible additional query parameters and
    Korp configuration parameters. It returns the JSON response for `/dependency-relations` with the given parameters
    (and cache=false).
    """

    def _dependency_relations_testcorpus(
        word: str, corpora: list[str] | str, params: dict | None = None, config: dict | None = None
    ) -> dict:
        query_params = {
            "corpora": make_liststr(corpora),
            "term": word,
            "cache": "false",
            "measures": "freq,mi",
        }
        database_tables(corpora, "dependency_relations")
        query_params.update(params or {})
        return get_json("/dependency-relations", params=query_params, config=config)

    return _dependency_relations_testcorpus


class TestDependencyRelations:
    """Tests for `/dependency-relations`."""

    @pytest.mark.parametrize("word", ["är"])
    @pytest.mark.parametrize("corpora", ["testcorpus2", ["testcorpus2", "testcorpus2b"]])
    @staticmethod
    def test_dependency_relations_simple(
        word: str, corpora: list[str] | str, dependency_relations_testcorpus: Callable[..., dict]
    ) -> None:
        """Test `/dependency-relations` with the given word and corpora."""
        data = dependency_relations_testcorpus(word, corpora)
        assert "relations" in data
        for wp in data["relations"]:
            assert wp["head"] == word or wp["dep"] == word

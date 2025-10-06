
"""
test_utils.py

Unit tests for functions in module korp.utils.
"""


from korp import utils


class TestParseCorpora:

    """Tests for parse_corpora"""

    def test_parse_corpora_empty(self):
        assert utils.parse_corpora({}) == []

    def test_parse_corpora_empty_string(self):
        assert utils.parse_corpora({"corpus": ""}) == [""]

    def test_parse_corpora_list(self):
        assert utils.parse_corpora({"corpus": ["A", "B"]}) == ["A", "B"]

    def test_parse_corpora_string(self):
        assert utils.parse_corpora(
            {"corpus": utils.QUERY_DELIM.join(["A", "B"])}) == ["A", "B"]

    def test_parse_corpora_string_upper(self):
        assert utils.parse_corpora(
            {"corpus": utils.QUERY_DELIM.join(["a", "b"])}) == ["A", "B"]

    def test_parse_corpora_sort(self):
        assert utils.parse_corpora({"corpus": ["B", "A"]}) == ["A", "B"]

    def test_parse_corpora_unique(self):
        assert (utils.parse_corpora({"corpus": ["A", "B", "A", "B"]}) ==
                ["A", "B"])

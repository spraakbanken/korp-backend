
"""
test_query.py

Pytest tests for the Korp /query endpoint
"""


from tests.testutils import get_response_json


class TestQuery:

    """Tests for /query"""

    def test_query_single_corpus(self, client, corpora):
        """Test a simple query on a single corpus."""
        data = get_response_json(
            client, "/query",
            query_string={
                "corpus": "testcorpus",
                "cqp": "[lemma=\"this\"]",
                "cache": "false",
            })
        kwic = data["kwic"]
        assert len(kwic) == data["hits"]
        # print(data)
        # assert 0

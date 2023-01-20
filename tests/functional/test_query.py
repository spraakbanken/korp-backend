
"""
test_query.py

Pytest tests for the Korp /query endpoint
"""


class TestQuery:

    """Tests for /query"""

    def test_query_single_corpus(self, client, corpora):
        """Test a simple query on a single corpus."""
        response = client.get(
            "/query",
            query_string={
                "corpus": "testcorpus",
                "cqp": "[lemma=\"this\"]",
                "cache": "false",
            })
        assert response.status_code == 200
        assert response.is_json == True
        data = response.get_json()
        kwic = data["kwic"]
        assert len(kwic) == data["hits"]
        print(data)
        # assert 0

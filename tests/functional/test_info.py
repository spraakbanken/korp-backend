
"""
test_info.py

Pytest tests for the Korp /info and /corpus_info endpoints.
"""


class TestInfo:

    """Tests for the /info endpoint"""

    def test_info_contains_version(self, client):
        """Test that /info response contains version info."""
        response = client.get("/info")
        assert response.status_code == 200
        assert response.is_json == True
        data = response.get_json()
        assert data["version"] and data["version"] != ""


class TestCorpusInfo:

    """Tests for the /corpus_info endpoint"""

    def test_corpus_info_single_corpus(self, client, corpora):
        """Test /corpus_info for a single corpus."""
        corpus = corpora[0].upper()
        response = client.get(
            "/corpus_info",
            query_string={
                "cache": "false",
                "corpus": corpus,
            })
        assert response.status_code == 200
        assert response.is_json == True
        data = response.get_json()
        print(data)
        corpus_data = data["corpora"][corpus]
        attrs = corpus_data["attrs"]
        # TODO: Add more specific assertions and perhaps split this
        # into multiple tests
        assert attrs
        assert attrs["p"]
        assert attrs["s"]
        assert data["total_size"]
        assert data["total_sentences"]

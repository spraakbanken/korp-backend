
"""
test_info.py

Pytest tests for the Korp /info endpoint.
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


"""
tests.testutils

Utility functions that can be called from tests. The functions may
contain assertions that are subject to rewriting.
"""


def get_response_json(client, *args, **kwargs):
    """Call client.get with given args, assert success, return response JSON."""
    # This function helps in making actual test functions for
    # endpoints slightly more compact and less repetitive
    response = client.get(*args, **kwargs)
    assert response.status_code == 200
    assert response.is_json
    return response.get_json()

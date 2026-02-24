
"""Utility functions that can be called from tests.

The functions may contain assertions that are subject to rewriting.
"""


from korp.utils import QUERY_DELIM


def get_response_json(client, *args, **kwargs) -> dict:
    """Call client.get with given args, assert success, return response JSON."""
    # This function helps in making actual test functions for
    # endpoints slightly more compact and less repetitive
    response = client.get(*args, **kwargs)
    assert response.status_code == 200
    assert "application/json" in response.headers.get("content-type", "")
    return response.json()


def make_liststr(arg: str | list[str]) -> str:
    """Return `arg` if it is a string, otherwise return a string joining the items of `arg` with `QUERY_DELIM`."""
    return arg if isinstance(arg, str) else QUERY_DELIM.join(arg)

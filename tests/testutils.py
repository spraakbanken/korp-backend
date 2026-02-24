"""Utility functions that can be called from tests.

The functions may contain assertions that are subject to rewriting.
"""

from typing import Any

from fastapi.testclient import TestClient

from korp.utils import QUERY_DELIM


def get_response_json(client: TestClient, *args: Any, **kwargs: Any) -> dict:
    """Call `client.get` with the given arguments and return the JSON content of the response.

    Args:
        client: The TestClient to use for the request.
        *args: Positional arguments to pass to `client.get`.
        **kwargs: Keyword arguments to pass to `client.get`.

    Returns:
        The JSON content of the response.
    """
    response = client.get(*args, **kwargs)
    assert response.status_code == 200  # noqa: PLR2004
    assert "application/json" in response.headers.get("content-type", "")
    return response.json()


def make_liststr(arg: str | list[str]) -> str:
    """Return `arg` if it is a string, otherwise return a string joining the items of `arg` with `QUERY_DELIM`."""
    return arg if isinstance(arg, str) else QUERY_DELIM.join(arg)


"""
tests.testutils

Utility functions that can be called from tests. The functions may
contain assertions that are subject to rewriting.
"""


from korp.utils import QUERY_DELIM


def get_response_json(client, *args, **kwargs):
    """Call client.get with given args, assert success, return response JSON."""
    # This function helps in making actual test functions for
    # endpoints slightly more compact and less repetitive
    response = client.get(*args, **kwargs)
    assert response.status_code == 200
    assert response.is_json
    return response.get_json()


def make_liststr(arg):
    """Return str arg as is, else return arg items separated by QUERY_DELIM."""
    if isinstance(arg, str):
        return arg
    else:
        return QUERY_DELIM.join(arg)

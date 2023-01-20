
"""
conftest.py

Pytest fixtures for testing the Korp backend as a Flask app.
"""


import pytest

from korp import create_app


@pytest.fixture()
def app():
    """Create and configure a Korp app instance."""
    app = create_app({
        # https://flask.palletsprojects.com/en/2.2.x/config/#TESTING
        "TESTING": True,
    })
    # print(app.config)
    yield app


@pytest.fixture()
def client(app):
    """Create and return a test client."""
    return app.test_client()

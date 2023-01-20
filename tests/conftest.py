
"""
conftest.py

Pytest fixtures for testing the Korp backend as a Flask app.
"""


import pytest

from pathlib import Path

from korp import create_app


@pytest.fixture(scope="session")
def corpus_data_root(tmp_path_factory):
    """Return a corpus data root directory for a session."""
    return tmp_path_factory.mktemp("corpora")


@pytest.fixture(scope="session")
def corpus_registry_dir(corpus_data_root):
    """Return a corpus registry directory for a session."""
    return str(corpus_data_root / "registry")


@pytest.fixture()
def app(corpus_registry_dir):
    """Create and configure a Korp app instance."""
    app = create_app({
        # https://flask.palletsprojects.com/en/2.2.x/config/#TESTING
        "TESTING": True,
        "CWB_REGISTRY": corpus_registry_dir,
    })
    # print(app.config)
    yield app


@pytest.fixture()
def client(app):
    """Create and return a test client."""
    return app.test_client()

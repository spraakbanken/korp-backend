
"""
conftest.py

Pytest fixtures for testing the Korp backend as a Flask app.
"""


import pytest

from pathlib import Path

from korp import create_app
from tests.corpusutils import CWBEncoder


# Functions in tests.utils are called by tests and contain assertions
# that should be rewritten
pytest.register_assert_rewrite("tests.testutils")


# Test data (source) directory
_datadir = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def corpus_data_root(tmp_path_factory):
    """Return a corpus data root directory for a session."""
    return tmp_path_factory.mktemp("corpora")


@pytest.fixture(scope="session")
def corpus_registry_dir(corpus_data_root):
    """Return a corpus registry directory for a session."""
    return str(corpus_data_root / "registry")


@pytest.fixture()
def cache_dir(tmp_path_factory):
    """Return a cache directory."""
    # Should this fixture have a non-default scope?
    return tmp_path_factory.mktemp("cache")


@pytest.fixture()
def corpus_config_dir(tmp_path_factory):
    """Return a corpus configuration directory."""
    # Should this fixture have a non-default scope (session?)?
    return tmp_path_factory.mktemp("corpus-config")


@pytest.fixture()
def app(corpus_registry_dir, cache_dir, corpus_config_dir):
    """Create and configure a Korp app instance."""
    app = create_app({
        # https://flask.palletsprojects.com/en/2.2.x/config/#TESTING
        "TESTING": True,
        "CWB_REGISTRY": corpus_registry_dir,
        "CACHE_DIR": cache_dir,
        "CORPUS_CONFIG_DIR": corpus_config_dir,
    })
    # print(app.config)
    yield app


@pytest.fixture()
def client(app):
    """Create and return a test client."""
    return app.test_client()


@pytest.fixture(scope="session")
def corpora(corpus_data_root):
    """Encode corpora in data/corpora/src and return their corpus ids."""
    corpus_source_dir = _datadir / "corpora" / "src"
    cwb_encoder = CWBEncoder(str(corpus_data_root))
    return cwb_encoder.encode_corpora(str(corpus_source_dir))

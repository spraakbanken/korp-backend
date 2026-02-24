"""Pytest fixtures for testing the Korp backend as a FastAPI app."""


import warnings
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from shutil import copytree

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from korp.app import create_app
from tests.corpusutils import CWBEncoder
from tests.dbutils import KorpDatabase

# Functions in tests.utils are called by tests and contain assertions
# that should be rewritten
pytest.register_assert_rewrite("tests.testutils")


# Test data (source) directory
_datadir = Path(__file__).parent / "data"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add pytest command-line options related to MySQL database access."""
    KorpDatabase.pytest_add_db_options(parser)


def pytest_configure(config: pytest.Config) -> None:
    """Process the command-line options related to MySQL database access."""
    KorpDatabase.pytest_config_db_options(config)


@pytest.fixture(scope="session")
def corpus_data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a corpus data root directory for a session."""
    return tmp_path_factory.mktemp("corpora")


@pytest.fixture(scope="session")
def corpus_registry_dir(corpus_data_root: Path) -> Path:
    """Return a corpus registry directory for a session."""
    return corpus_data_root / "registry"


@pytest.fixture
def cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a cache directory."""
    return tmp_path_factory.mktemp("cache")


@pytest.fixture(scope="session")
def corpus_config_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a corpus configuration directory."""
    return tmp_path_factory.mktemp("corpus-config")


@pytest.fixture(scope="session")
def _database() -> Iterator[KorpDatabase]:
    """Create and yield a KorpDatabase (Korp MySQL database) for a session.

    If the database could not be created, the dbname attribute of the returned value is `None`. Afterwards, drop the
    database.

    Actual tests should use the fixture `database` (or even better, `database_tables`) instead of this, so that they are
    skipped if the test database cannot be created.

    Yields:
        A KorpDatabase instance.
    """
    db = KorpDatabase(_datadir / "db")
    db.create()
    yield db
    db.drop()


@pytest.fixture(scope="session")
def database(_database: KorpDatabase) -> KorpDatabase:
    """Return a KorpDatabase for a session; if that fails, skip test."""
    if _database.dbname is None:
        error = _database.create_error
        msg = ""
        if error is not None:
            msg = "Unable to create Korp database: Error " + error["message"]
            if error["sql"] is not None:
                msg += " when executing SQL statement: " + error["sql"]
            warnings.warn(f"Skipping tests using Korp database: {msg}")
        pytest.skip(msg)
    return _database


@pytest.fixture
def database_tables(database: KorpDatabase) -> Callable[..., None]:
    """Return a function for importing Korp database tables.

    The returned function takes as its arguments a list of corpora
    (corpus ids) or a single corpus id (string) whose data to import,
    and the type of table data to import (if omitted, import all
    types). The function drops possibly existing tables, so all the
    tables for a test should be imported with a single call.
    """

    def _database_tables(corpora: str | list[str], table_type: str | None = None) -> None:
        """Import Korp database tables of table_type for corpora."""
        database.import_tables(corpora, table_type)

    return _database_tables


@pytest.fixture
def app_factory(
    corpus_registry_dir: Path, cache_dir: Path, corpus_config_dir: Path, _database: KorpDatabase
) -> Callable[..., FastAPI]:
    """Return a function creating and configuring a Korp app instance."""

    def _app_factory(config: dict | None = None) -> FastAPI:
        """Return Korp app instance with config overriding defaults."""
        base_config = {
            "TESTING": True,
            "CWB_REGISTRY": str(corpus_registry_dir),
            "CACHE_DIR": str(cache_dir),
            "CORPUS_CONFIG_DIR": str(corpus_config_dir),
        }
        # Update the configuration from the database configuration, as
        # custom pytest command-line options can be used to change the
        # MySQL connection parameters
        base_config.update(_database.get_config())
        base_config.update(config or {})
        return create_app(base_config)

    return _app_factory


@pytest.fixture
def client(app_factory: Callable[..., FastAPI]) -> Iterator[TestClient]:
    """Yield a TestClient for a default app instance created with app_factory."""
    with TestClient(app_factory()) as test_client:
        yield test_client


@pytest.fixture
def client_factory(app_factory: Callable[..., FastAPI]) -> Callable[..., AbstractContextManager[TestClient]]:
    """Return a context manager creating a TestClient for custom config."""

    @contextmanager
    def _client_factory(config: dict | None = None) -> Iterator[TestClient]:
        with TestClient(app_factory(config or {})) as test_client:
            yield test_client

    return _client_factory


@pytest.fixture
def get_json(client_factory: Callable[..., AbstractContextManager[TestClient]]) -> Callable[..., dict]:
    """Return helper for GET requests returning validated JSON."""

    def _get_json(path: str, *, params: dict | None = None, config: dict | None = None):
        from tests.testutils import get_response_json

        with client_factory(config or {}) as test_client:
            return get_response_json(test_client, path, params=params)

    return _get_json


@pytest.fixture(scope="session")
def corpora(corpus_data_root):
    """Encode corpora in data/corpora/src and return their corpus ids."""
    corpus_source_dir = _datadir / "corpora" / "src"
    cwb_encoder = CWBEncoder(str(corpus_data_root))
    return cwb_encoder.encode_corpora(str(corpus_source_dir))


@pytest.fixture
def corpus_configs(corpus_config_dir):
    """Copy corpus configs from data/corpora/config to corpus_config_dir."""
    config_src_dir = _datadir / "corpora" / "config"
    copytree(str(config_src_dir), str(corpus_config_dir), dirs_exist_ok=True)

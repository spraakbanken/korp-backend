"""Configuration settings for Korp backend.

These settings can be overridden by environment variables or by creating a .env file in the directory where you start
the server.
"""

from pathlib import Path
from types import UnionType
from typing import Any, Literal, get_args, get_origin

import yaml
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for Korp backend."""

    # Host and port for the WSGI server
    WSGI_HOST: str = "0.0.0.0"
    WSGI_PORT: int = 8000

    # Server instances used in API documentation
    # E.g. [{"url": "http://localhost:8000", "description": "Local development server"}]
    SERVERS: list[dict[str, str]] = []

    # The absolute path to the CQP binaries (required)
    CQP_EXECUTABLE: Path | None = None
    CWB_SCAN_EXECUTABLE: Path | None = None

    # The absolute path to the CWB registry files (required)
    CWB_REGISTRY: Path | None = None

    # The default encoding for the cqp binary
    CQP_ENCODING: str = "UTF-8"

    # Locale to use when sorting
    LC_COLLATE: str = "sv_SE.UTF-8"

    # The maximum number of search results that can be returned per query (0 = no limit)
    MAX_KWIC_ROWS: int = 0

    # Number of threads to use during parallel processing
    PARALLEL_THREADS: int = 3

    # Database host and port
    DB_HOST: str = "0.0.0.0"
    DB_PORT: int = 3306

    # Database name
    DB_NAME: str = ""

    # Database character set (use "utf8mb4" for full Unicode)
    DB_CHARSET: str = "utf8"

    # Dependency relations table prefix
    DB_DEPENDENCY_RELATIONS_TABLE_PREFIX: str = "relations"

    # Lexeme counts table name
    DB_LEXEME_COUNTS_TABLE: str = "lexeme_counts"

    # Username and password for database access
    DB_USER: str = ""
    DB_PASSWORD: str = ""

    # Database connection timeout in seconds
    DB_CONNECT_TIMEOUT: int = 10

    # Database read timeout in seconds
    DB_READ_TIMEOUT: int = 60

    # Time to wait for a connection from the database connection pool before raising an error (in seconds)
    DB_POOL_TIMEOUT: int = 30

    # Log database queries that take longer than this many seconds to execute (0 = disable)
    DB_SLOW_QUERY_SECONDS: float = 30.0

    # Max length of SQL statements in logs (0 = no limit)
    DB_LOG_SQL_MAX_LENGTH: int = 300

    # HTTP Cache-Control header max-age value (in hours)
    HTTP_CACHE_MAXAGE: int = 1

    # Log requests that take longer than this many seconds to complete (0 = disable)
    REQUEST_SLOW_LOG_SECONDS: float = 0.0

    # If REQUEST_SLOW_LOG_SECONDS is enabled, keep logging the request at this interval until it completes
    REQUEST_STUCK_LOG_INTERVAL_SECONDS: float = 60.0

    # Minimum number of rows for timespan calculation to use a separate process (0 = always use thread)
    TIMESPAN_PROCESS_THRESHOLD_ROWS: int = 100_000

    # Maximum number of rows to cache for timespan calculations (0 = no caching)
    TIMESPAN_CACHE_MAX_ROWS: int = 50_000

    # Log timespan calculation phases if total duration exceeds this threshold (in seconds, 0 = disable)
    TIMESPAN_PHASE_LOG_SECONDS: float = 0.0

    # Cache path (optional). Script must have read and write access.
    CACHE_DIR: Path | None = None

    # Disk cache lifespan in minutes
    CACHE_LIFESPAN: int = 20

    # Memcached server IP address and port. Sockets are not supported.
    MEMCACHED_SERVER: str | None = None

    # Enable route-specific rate limiting
    RATE_LIMIT_ENABLED: bool = False

    # Optional explicit storage URI for rate limiting. If unset and rate limiting is enabled, MEMCACHED_SERVER is used.
    RATE_LIMIT_STORAGE_URI: str | None = None

    # When to include rate limit headers in responses (`none`, `on_reject`, `always`)
    RATE_LIMIT_HEADERS: Literal["none", "on_reject", "always"] = "on_reject"

    # Default rate limit applied to all rate-limited routes (marked with `@api_handler(rate_limit=True)`). Uses `limits`
    # syntax, e.g. "10/minute" or "1/second;60/minute". Empty string means no limit.
    RATE_LIMIT_DEFAULT: str = ""

    # Per-route rate limit overrides, keyed by route path (e.g. {"query": "30/minute"}). Overrides
    # RATE_LIMIT_DEFAULT for matched routes. Empty value disables the rate limit for that route. Paths are matched
    # with or without leading slash.
    RATE_LIMITS: dict[str, str] = {}

    # Max number of rows from count command to cache
    CACHE_MAX_STATS: int = 50

    # Max size in bytes per cached query data file (0 = no limit)
    CACHE_MAX_QUERY_DATA: int = 0

    # Corpus configuration directory
    CORPUS_CONFIG_DIR: Path | None = None

    # Set to True to enable "lab mode", potentially enabling experimental features and access to lab-only corpora
    LAB_MODE: bool = False

    # Plugins to load
    PLUGINS: list = []

    # Plugin configuration file
    PLUGINS_CONFIG_FILE: Path = Path("plugins.yaml")

    # Optional inline plugin configuration overrides
    PLUGINS_CONFIG: dict = {}

    # This is set to True automatically when the app is run in test mode, and can be used by plugins and other code to
    # conditionally enable test-specific behavior. It is not intended to be set manually.
    TESTING: bool = False

    # Override default settings with settings from .env file
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_nested_delimiter="__", env_parse_none_str="null"
    )

    @model_validator(mode="after")
    def _expand_paths(self) -> "Settings":
        """Expand `~` in all Path fields.

        Returns:
            The settings instance with expanded paths.
        """
        for name, field_info in type(self).model_fields.items():
            annotation = field_info.annotation
            # Match both `Path` and `Path | None`
            types = get_args(annotation) if get_origin(annotation) is UnionType else (annotation,)
            if Path in types:
                value = getattr(self, name)
                if isinstance(value, Path):
                    setattr(self, name, value.expanduser())
        return self

    @staticmethod
    def _load_plugins_config_file(file_path: Path) -> dict[str, Any]:
        """Load plugin configuration from YAML file.

        Args:
            file_path: Path to YAML file.

        Returns:
            A plugin configuration mapping.

        Raises:
            ValueError: If the YAML root is not a mapping.
        """
        path = file_path.expanduser()
        if not path.is_file():
            return {}

        with path.open(encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}

        if not isinstance(loaded, dict):
            msg = f"PLUGINS_CONFIG_FILE must contain a YAML mapping, got {type(loaded).__name__}"
            raise ValueError(msg)

        return loaded

    @model_validator(mode="before")
    @classmethod
    def _merge_plugins_config_file(cls, data: Any) -> Any:
        """Merge plugin config from YAML file with explicitly provided plugin config.

        Args:
            data: Raw settings data before model validation.

        Returns:
            Updated settings data with merged plugin configuration.

        Raises:
            ValueError: If the plugin config file cannot be read or parsed.
        """
        if not isinstance(data, dict):
            return data

        plugins_config_file = data.get("PLUGINS_CONFIG_FILE", cls.model_fields["PLUGINS_CONFIG_FILE"].default)
        if not plugins_config_file:
            return data

        try:
            file_config = cls._load_plugins_config_file(plugins_config_file)
        except (OSError, yaml.YAMLError) as error:
            msg = f"Could not read PLUGINS_CONFIG_FILE '{plugins_config_file}': {error}"
            raise ValueError(msg) from error

        explicit_config = data.get("PLUGINS_CONFIG") or {}
        if not isinstance(explicit_config, dict):
            return data

        data["PLUGINS_CONFIG"] = {**file_config, **explicit_config}
        return data


settings = Settings()

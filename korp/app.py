"""Application factory for the Korp backend FastAPI app."""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from logging import getLogger
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from korp import auth, caching, handler
from korp.api.routers import routers
from korp.config import Settings, settings
from korp.cwb import CWB
from korp.db import MySQL
from korp.memcached import Memcached

logger = getLogger(__name__)

_TAGS = [
    {
        "name": "Corpus Information",
        "description": "Routes for retrieving information about corpora and their attributes.",
    },
    {
        "name": "Concordance",
        "description": "Routes for retrieving concordance lines and related information.",
    },
    {"name": "Statistics", "description": "Routes for retrieving various corpus statistics."},
    {
        "name": "Word Relations",
        "description": "Routes for querying word relations.",
    },
    {
        "name": "Administration",
        "description": "Routes for administrative tasks.",
    },
]


def _apply_settings_override(config_override: dict[str, Any] | None) -> bool:
    """Apply config overrides to the global settings object.

    If `TESTING` is set to `True` in the overrides, disables plugins unless explicitly specified.

    Returns:
        Whether this app run should be considered test mode.
    """
    if not config_override:
        return False

    testing = bool(config_override.get("TESTING", False))
    settings_overrides: dict[str, Any] = {}

    for config_key, value in config_override.items():
        if config_key not in Settings.model_fields:
            # Warn about unknown config keys
            logger.warning("Unknown config key in override: %s", config_key)
            continue
        settings_overrides[config_key] = value

    if testing and "PLUGINS" not in settings_overrides:
        settings_overrides["PLUGINS"] = []
        settings_overrides.setdefault("PLUGINS_CONFIG", {})

    if not settings_overrides:
        return testing

    merged = settings.model_dump()
    merged.update(settings_overrides)
    validated = Settings(**merged)
    for key, value in validated.model_dump().items():
        setattr(settings, key, value)

    return testing


def _get_required_cwb_settings() -> tuple[Path, Path, Path]:
    """Get required CWB settings and raise an error if any are missing.

    Returns:
        A tuple with CQP executable path, cwb-scan-corpus executable path, and CWB registry path.

    Raises:
        RuntimeError: If any required CWB settings are missing.
    """
    executable = settings.CQP_EXECUTABLE
    scan_executable = settings.CWB_SCAN_EXECUTABLE
    registry = settings.CWB_REGISTRY

    if executable is None or scan_executable is None or registry is None:
        missing = [
            name
            for name, val in [
                ("CQP_EXECUTABLE", executable),
                ("CWB_SCAN_EXECUTABLE", scan_executable),
                ("CWB_REGISTRY", registry),
            ]
            if val is None
        ]
        raise RuntimeError(f"Missing required settings: {', '.join(missing)}")

    return executable, scan_executable, registry


def create_app(config_override: dict[str, Any] | None = None) -> FastAPI:
    """Create and configure a FastAPI app instance.

    Args:
        config_override: Optional mapping of settings overrides using `Settings` field names.

    Returns:
        A configured FastAPI app instance.

    Raises:
        ImportError: If a plugin specified in the settings cannot be imported.
        TypeError: If a plugin exports an invalid authorizer class.
        RuntimeError: If more than one plugin exports an authorizer class.
    """
    testing = _apply_settings_override(config_override)
    cqp_executable, cwb_scan_executable, cwb_registry = _get_required_cwb_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Initialize and finalize resources for the app lifespan."""
        try:
            handler.enforce_ctx_dependency(app)
            app.state.cwb = CWB(
                executable=cqp_executable,
                scan_executable=cwb_scan_executable,
                registry=cwb_registry,
                locale=settings.LC_COLLATE,
                encoding=settings.CQP_ENCODING,
            )

            app.state.db.init(settings)

            await app.state.memcached.init(settings.MEMCACHED_SERVER)
            app.state.cache_enabled = (
                app.state.memcached.active and settings.CACHE_DIR and Path(settings.CACHE_DIR).is_dir()
            )
            if app.state.cache_enabled:
                logger.info("Caching is enabled.")
                await caching.setup_cache(app.state.memcached)
            else:
                logger.info("Caching is disabled.")
            authorizer_class = getattr(app.state, "authorizer_class", None)
            if authorizer_class:
                app.state.authorizer = authorizer_class(app.state.cwb, app.state.memcached)
            else:
                app.state.authorizer = None
            yield
        finally:
            # Clean up resources on shutdown or if initialization fails
            app.state.authorizer = None
            await app.state.memcached.close()
            await app.state.db.close()

    app = FastAPI(
        title="Korp Backend",
        version=importlib.metadata.version("korp-backend"),
        lifespan=lifespan,
        openapi_tags=_TAGS,
    )

    app.state.db = MySQL()
    app.state.memcached = Memcached()
    app.state.testing = testing

    @app.middleware("http")
    async def support_post_params(request: Request, call_next: Any) -> Any:
        """Convert POST body parameters to query parameters.

        This allows clients to send parameters in the body of POST requests instead of the URL query string, to
        avoid issues with URL length limits.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler to call.

        Returns:
            The response from the next handler.
        """
        await handler.convert_post_body_to_query_params(request)
        return await call_next(request)

    # Enable CORS, with support for credentials and an Access-Control-Max-Age (for preflight requests)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=settings.HTTP_CACHE_MAXAGE * 3600,
    )

    for router in routers:
        app.include_router(router)

    # Load plugins
    authorizer_class: type[auth.Authorizer] | None = None
    authorizer_plugin: str | None = None
    for plugin in settings.PLUGINS:
        try:
            module = importlib.import_module(plugin)
        except ImportError:
            logger.warning("Failed to import plugin %s", plugin)
            raise
        # Find all routers defined in plugin module and register them
        for name in dir(module):
            value = getattr(module, name)
            if isinstance(value, APIRouter):
                app.include_router(value)

        if (plugin_authorizer := getattr(module, "AUTHORIZER_CLASS", None)) is None:
            continue
        if not inspect.isclass(plugin_authorizer) or not issubclass(plugin_authorizer, auth.Authorizer):
            raise TypeError(
                f"Plugin {plugin} exports AUTHORIZER_CLASS={plugin_authorizer!r}, "
                "but it is not a subclass of Authorizer."
            )
        if authorizer_class is not None:
            raise RuntimeError(
                "Multiple auth plugins export AUTHORIZER_CLASS: "
                f"{authorizer_plugin} and {plugin}. Configure only one authorizer plugin."
            )
        authorizer_class = plugin_authorizer
        authorizer_plugin = plugin

    app.state.authorizer_class = authorizer_class
    app.state.authorizer = None

    return app

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
from korp.rate_limit import RequestRateLimiter, resolve_rate_limit_storage_uri

logger = getLogger(__name__)

# Structure for API documentation, defining tags and route order
_API_DOCUMENTATION_STRUCTURE = [
    {
        "name": "Corpus Information",
        "description": "Routes for retrieving corpus metadata and configuration.",
        "routes": ["/info", "/corpora/info", "/attribute_values", "/corpora/config"],
    },
    {
        "name": "Concordance",
        "description": "Routes for retrieving concordance lines and related information.",
        "routes": ["/concordance", "/concordance/sample"],
    },
    {
        "name": "Statistics",
        "description": "Routes for retrieving various corpus statistics.",
        "routes": [
            "/frequencies",
            "/frequencies/corpus",
            "/frequencies/time",
            "/token_distribution",
            "/log_likelihood",
            "/lexeme_counts",
        ],
    },
    {
        "name": "Dependency Relations",
        "description": "Routes for querying dependency relation statistics.",
        "routes": [
            "/dependency_relations",
            "/dependency_relations/time",
            "/dependency_relations/sentences",
            "/dependency_relations/time/sentences",
        ],
    },
    {
        "name": "Administration",
        "description": "Routes for administrative tasks.",
    },
]

_OPENAPI_TAGS = [
    {k: v for k, v in tag.items() if k != "routes"}
    for tag in _API_DOCUMENTATION_STRUCTURE
]

_DESCRIPTION = """
# Korp Backend API

Korp is a corpus search system developed at [Språkbanken Text](https://spraakbanken.gu.se/eng). The Korp backend API
lets applications query annotated text corpora, inspect corpus metadata, and calculate corpus statistics. It powers the
[Korp frontend](https://github.com/spraakbanken/korp-frontend), and can also be used directly by other applications and
scripts.

Use this API to:

- discover available corpora and their annotations
- run concordance queries and retrieve matching lines with annotations
- frequency query matches, annotation values, and time distributions
- query dependency relation statistics when relation data is configured
- maintain server-side caches

Available corpora, annotations, and optional features depend on the Korp installation you are using.

The [source code](https://github.com/spraakbanken/korp-backend) is available on GitHub under the MIT license.

## Request Basics

A typical API request is an HTTP `GET` request following the pattern:

> `/command?parameter=value&parameter=value...`

Parameters are typically sent as query parameters. Parameters that accept multiple values usually support both
comma-separated values and repeated parameters.

While the API documentation only presents endpoints as accepting `GET` requests with query parameters, the backend also
supports `POST` requests with parameters sent in the request body, encoded as either `application/x-www-form-urlencoded`
or `application/json`. This makes it possible to send long queries that might otherwise exceed URL length limits. If a
parameter is sent both in the query string and the body, the query string value takes precedence.

All responses are returned as JSON.

## Return Codes

Most API routes stream the response body. Streaming lets the backend send keepalive whitespace while long-running CQP or
database work is still in progress, which helps avoid proxy and browser timeouts.

Because HTTP status and headers are sent before the full result has been computed, errors that happen after streaming
has started cannot be reported by changing the HTTP status code. In those cases the response still has status `200`,
and the JSON object contains an `error` field with the error details. Clients should therefore check the response body
for `error` instead of treating HTTP `200` alone as a successful API result. When `debug=true`, errors may include extra
debug information such as tracebacks.

Errors that happen before the route starts streaming may still use normal HTTP status codes, for example malformed
requests, unknown routes, validation errors, or rate limiting.

## CQP Queries

For many routes, Korp uses Corpus Workbench's CQP query language. For details about the query syntax, see the [CQP Query
Language Tutorial](http://cwb.sourceforge.net/files/CQP_Tutorial.pdf).
"""

_CONTACT = {
    "name": "Språkbanken Text",
    "url": "https://spraakbanken.gu.se/eng",
    "email": "sb-info@svenska.gu.se",
}

_LICENSE = {
    "name": "MIT License",
    "identifier": "MIT",
    "url": "https://opensource.org/licenses/MIT",
}


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
            if settings.CACHE_DIR and not Path(settings.CACHE_DIR).is_dir():
                logger.warning(
                    "Cache directory %s does not exist or is not a directory. Caching will be disabled.",
                    settings.CACHE_DIR,
                )
            if app.state.cache_enabled:
                logger.info("Caching is enabled.")
                await caching.setup_cache(app.state.memcached)
            else:
                logger.info("Caching is disabled.")

            app.state.rate_limiter = None
            if settings.RATE_LIMIT_ENABLED:
                storage_uri = resolve_rate_limit_storage_uri(settings)
                if storage_uri:
                    try:
                        app.state.rate_limiter = await RequestRateLimiter.create(
                            storage_uri,
                            headers_mode=settings.RATE_LIMIT_HEADERS,
                        )
                        logger.info("Rate limiting is enabled.")
                    except Exception:
                        logger.warning(
                            "Rate limiting is enabled but the storage backend is unavailable. "
                            "Rate limiting will be disabled.",
                            exc_info=True,
                        )
                else:
                    logger.warning(
                        "Rate limiting is enabled but no storage is configured "
                        "(set RATE_LIMIT_STORAGE_URI or MEMCACHED_SERVER). Rate limiting will be disabled."
                    )
            else:
                logger.info("Rate limiting is disabled.")

            authorizer_class = getattr(app.state, "authorizer_class", None)
            if authorizer_class:
                app.state.authorizer = authorizer_class(app.state.cwb, app.state.memcached)
            else:
                app.state.authorizer = None
            yield
        finally:
            # Clean up resources on shutdown or if initialization fails
            app.state.authorizer = None
            if app.state.rate_limiter is not None:
                await app.state.rate_limiter.close()
                app.state.rate_limiter = None
            await app.state.memcached.close()
            await app.state.db.close()

    app = FastAPI(
        title="Korp Backend",
        summary="API backend for Korp, a corpus query system.",
        description=_DESCRIPTION,
        version=importlib.metadata.version("korp-backend"),
        lifespan=lifespan,
        openapi_tags=_OPENAPI_TAGS,
        contact=_CONTACT,
        license_info=_LICENSE,
        servers=settings.SERVERS,
    )

    app.state.db = MySQL()
    app.state.memcached = Memcached()
    app.state.testing = testing
    app.state.rate_limiter = None

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
        allow_origins=settings.CORS_ALLOW_ORIGINS,
        allow_origin_regex=settings.CORS_ALLOW_ORIGIN_REGEX,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
        max_age=settings.HTTP_CACHE_MAXAGE * 3600,
    )

    # Save original OpenAPI method
    original_openapi = app.openapi

    def customize_openapi_response() -> dict[str, Any]:
        """Customize the OpenAPI schema response.

        This puts common response properties at the end of response schemas and removes auto-generated titles from
        parameters.

        Returns:
            The customized OpenAPI schema.
        """
        if app.openapi_schema is not None:
            return app.openapi_schema
        schema = original_openapi()

        # Reorder response properties
        common_keys = ("elapsed", "debug", "error")
        comps = schema.get("components", {}).get("schemas", {})
        for s in comps.values():
            props = s.get("properties")
            if not props:
                continue
            # Keep original order for non-common, then append common keys in the order defined in common_keys
            new_props = {k: v for k, v in props.items() if k not in common_keys}
            for k in common_keys:
                if k in props:
                    new_props[k] = props[k]
            s["properties"] = new_props

        # Remove auto-generated titles from parameters
        for path_item in schema.get("paths", {}).values():
            for operation in path_item.values():
                if not isinstance(operation, dict):
                    continue
                for parameter in operation.get("parameters", []):
                    if isinstance(parameter, dict):
                        parameter.get("schema", {}).pop("title", None)

        # Reorder routes according to _API_DOCUMENTATION_STRUCTURE
        routes_order = [r for t in _API_DOCUMENTATION_STRUCTURE for r in t.get("routes", [])]
        new_paths = {}
        for route in routes_order:
            new_paths[route] = schema["paths"][route]
        for route, path_item in schema["paths"].items():
            if route not in new_paths:
                new_paths[route] = path_item
        schema["paths"] = new_paths

        app.openapi_schema = schema
        return schema

    app.openapi = customize_openapi_response

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

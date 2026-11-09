"""Optional Matomo request tracking.

Install the `matomo` extra and enable `plugins.matomo` in `PLUGINS`. Configure middleware options in the
`plugins.matomo` entry of `PLUGINS_CONFIG` or the YAML file referenced by `PLUGINS_CONFIG_FILE`.
"""

from asgi_matomo import MatomoMiddleware
from fastapi import FastAPI

from korp.config import settings


def setup(app: FastAPI) -> None:
    """Register Matomo middleware with this plugin's configuration.

    Args:
        app: Application being configured.
    """
    app.add_middleware(MatomoMiddleware, **settings.PLUGINS_CONFIG[__name__])

"""An example plugin for Korp.

Enable and configure by updating the Korp configuration as follows:

PLUGINS = ["plugins.example"]

Add the following to the plugin configuration file specified by PLUGINS_CONFIG_FILE (default: plugins.yaml):

plugins.example:
    greeting: "Hello!"
"""

from korp import plugin
from korp.dependencies import CtxDep
from korp.handler import api_handler

router = plugin.Plugin("example", __name__)


@router.get("/hello")
@api_handler
def hello(_ctx: CtxDep) -> dict:
    """Return a greeting message from the plugin configuration."""
    return {"message": router.config("greeting", "Greeting not set")}

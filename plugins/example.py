"""An example plugin for Korp.

Enable and configure by updating the Korp configuration as follows:

PLUGINS = ["plugins.example"]

Add the following to the plugin configuration file specified by PLUGINS_CONFIG_FILE (default: plugins.yaml):

plugins.example:
    greeting: "Hello!"
"""

from korp import utils

router = utils.Plugin("example", __name__)


@router.get("/hello")
@utils.api_handler
def hello(_ctx: utils.CtxDep) -> dict:
    """Return a greeting message from the plugin configuration."""
    return {"message": router.config("greeting", "Greeting not set")}

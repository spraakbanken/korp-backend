"""An example plugin for Korp.

Enable and configure in `.env`:

PLUGINS = ["plugins.example"]
PLUGINS_CONFIG = '{
  "plugins.example": {
    "greeting": "Hello!"
  }
}'

"""

from korp import plugin
from korp.dependencies import CtxDep
from korp.handler import api_handler

router = plugin.Plugin("example", __name__)


@router.get("/hello", operation_id="get_hello")
@api_handler
def hello(_ctx: CtxDep) -> dict:
    """Return a greeting message from the plugin configuration."""
    return {"message": router.config("greeting", "Greeting not set")}

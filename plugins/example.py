"""An example plugin for Korp.

Enable and configure by updating the Korp configuration as follows:

PLUGINS = ["plugins.example"]

PLUGINS_CONFIG = {
    "plugins.example": {
        "greeting": "Hello!"
    }
}
"""

from korp.utils import main_handler, Plugin

bp = Plugin("example", __name__)


@bp.route("/hello")
@main_handler
def hello(_args):
    yield {
        "message": bp.config("greeting", "Greeting not set")
    }

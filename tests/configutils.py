
"""
tests/configutils.py

Utility functions used in pytest tests for Korp, in particular for
handling the Korp configuration.
"""


from importlib import import_module
from pathlib import Path

from flask.config import Config


def get_korp_config():
    """Return the Korp configuration as a `flask.config.Config` object.

    Return the Korp configuration from module `instance.config`, with
    defaults from module `config` for variables not defined in
    `instance.config`.

    Note that this assumes that the instance configuration file is the
    default one (`"{root_path}/instance/config.py"`).
    """
    # Find Korp package root based on the location of the config
    # module
    korp_dir = Path(import_module("config").__file__).parent
    conf = Config(str(korp_dir))
    # Load the default configuration
    conf.from_object("config")
    # Try to load the instance configuration
    instance_config_path = korp_dir / 'instance' / 'config.py'
    if instance_config_path.is_file():
        conf.from_pyfile(str(instance_config_path))
    return conf

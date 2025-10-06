
"""
tests/configutils.py

Utility functions used in pytest tests for Korp, in particular for
handling the Korp configuration.
"""


def get_korp_config():
    """Return the Korp configuration as a dict.

    Return the Korp configuration fron module instance.config, or if
    that is not available, from module config.
    """
    try:
        from instance import config
    except ImportError:
        import config
    # Treat all uppercase items in the config module as configuration
    # variables and add them to the dict to return
    return dict((key, val) for key, val in config.__dict__.items()
                if key.isupper())

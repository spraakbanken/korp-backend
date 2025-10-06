import importlib
from pathlib import Path

from flask import Flask
from flask.blueprints import Blueprint
from flask_cors import CORS

from korp import utils
from korp.cwb import cwb
from korp.db import mysql
from korp.memcached import memcached

# The version of this script
__version__ = "8.2.5"


def create_app(config_override=None):
    """Application factory, creating and configuring the Flask app."""

    app = Flask(__name__)

    # Enable CORS
    CORS(app, supports_credentials=True)

    # Load default config
    app.config.from_object("config")

    # Overwrite with instance config
    instance_config_path = Path(app.instance_path) / "config.py"
    if instance_config_path.is_file():
        app.config.from_pyfile(str(instance_config_path))
    else:
        print(f"Configure Korp by copying config.py to '{app.instance_path}' and modifying that copy")

    if config_override is not None:
        app.config.update(config_override)

    cwb.init(
        executable=app.config["CQP_EXECUTABLE"],
        scan_executable=app.config["CWB_SCAN_EXECUTABLE"],
        registry=app.config["CWB_REGISTRY"],
        locale=app.config["LC_COLLATE"],
        encoding=app.config["CQP_ENCODING"]
    )

    # Configure database connection
    app.config.from_mapping(
        MYSQL_HOST=app.config["DBHOST"],
        MYSQL_USER=app.config["DBUSER"],
        MYSQL_PASSWORD=app.config["DBPASSWORD"],
        MYSQL_DB=app.config["DBNAME"],
        MYSQL_PORT=app.config["DBPORT"],
        MYSQL_USE_UNICODE=True,
        MYSQL_CURSORCLASS="DictCursor",
    )

    mysql.init_app(app)

    # Set up Memcached
    if app.config["MEMCACHED_SERVER"]:
        memcached.init(app.config["MEMCACHED_SERVER"])
    app.config["CACHE_DISABLED"] = not memcached.active

    # Set up caching
    if not app.config["CACHE_DISABLED"]:
        with app.app_context():
            utils.setup_cache()

    # Register blueprints
    from .views import (
        cache,
        corpus_config,
        count,
        info,
        lemgram_count,
        loglike,
        misc,
        query,
        relations,
        attr_values,
        timespan
    )
    app.register_blueprint(cache.bp)
    app.register_blueprint(corpus_config.bp)
    app.register_blueprint(count.bp)
    app.register_blueprint(info.bp)
    app.register_blueprint(lemgram_count.bp)
    app.register_blueprint(loglike.bp)
    app.register_blueprint(misc.bp)
    app.register_blueprint(query.bp)
    app.register_blueprint(relations.bp)
    app.register_blueprint(attr_values.bp)
    app.register_blueprint(timespan.bp)

    # Load plugins
    for plugin in app.config["PLUGINS"]:
        module = importlib.import_module(plugin)
        # Find all blueprints defined in module and register them
        for name in dir(module):
            v = getattr(module, name)
            if isinstance(v, Blueprint):
                app.register_blueprint(v)

    # Register authorizer
    if utils.Authorizer.auth_class:
        utils.authorizer = utils.Authorizer.auth_class()

    return app

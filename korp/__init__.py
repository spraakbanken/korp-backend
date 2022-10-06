from pathlib import Path

try:
    import pylibmc
    cache_disabled = False
except ImportError:
    print("Could not load pylibmc. Caching will be disabled.")
    cache_disabled = True

from flask import Flask
from flask_cors import CORS

from korp import utils
from korp.cwb import cwb
from korp.db import mysql
from korp.memcached import memcached

# The version of this script
VERSION = "8.2.0"


def create_app():
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

    # Set up Memcached client pool
    global cache_disabled
    if app.config["MEMCACHED_SERVERS"] and not cache_disabled:
        memcached.init(app.config["MEMCACHED_SERVERS"], app.config["MEMCACHED_POOL_SIZE"])
    app.config["CACHE_DISABLED"] = True if cache_disabled or memcached.pool is None else False

    # Set up caching
    if not app.config["CACHE_DISABLED"]:
        with app.app_context():
            utils.setup_cache()

    # Register blueprints
    from .views import (
        authenticate,
        cache,
        corpus_config,
        count,
        info,
        lemgram_count,
        loglike,
        misc,
        query,
        relations,
        struct_values,
        timespan
    )
    app.register_blueprint(authenticate.bp)
    app.register_blueprint(cache.bp)
    app.register_blueprint(corpus_config.bp)
    app.register_blueprint(count.bp)
    app.register_blueprint(info.bp)
    app.register_blueprint(lemgram_count.bp)
    app.register_blueprint(loglike.bp)
    app.register_blueprint(misc.bp)
    app.register_blueprint(query.bp)
    app.register_blueprint(relations.bp)
    app.register_blueprint(struct_values.bp)
    app.register_blueprint(timespan.bp)

    return app

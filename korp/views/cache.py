import glob
import os
import time

from flask import Blueprint
from flask import current_app as app

from korp import utils
from korp.memcached import memcached

bp = Blueprint("cache", __name__)


@bp.route("/cache", methods=["GET", "POST"])
@utils.main_handler
@utils.prevent_timeout
def cache_handler(_args):
    """Check for updated corpora and invalidate caches where needed. Also remove old disk cache."""
    if not app.config["CACHE_DIR"] or not app.config["MEMCACHED_SERVER"] or app.config["CACHE_DISABLED"]:
        return {}

    result = {}

    # Set up caching if needed
    initial_setup = utils.setup_cache()

    if initial_setup:
        result["initial_setup"] = True
    else:
        result = {"multi_invalidated": False,
                  "multi_config_invalidated": False,
                  "corpora_invalidated": 0,
                  "configs_invalidated": 0,
                  "files_removed": 0}
        now = time.time()

        # Get modification time of corpus registry files
        corpora = utils.get_corpus_timestamps()
        # Get modification time of corpus config files
        corpora_configs, config_modes, config_presets = utils.get_corpus_config_timestamps()

        memcached_data = {}

        with memcached.get_client() as mc:

            last_update_keys = {corpus: f"{corpus}:last_update" for corpus in corpora}
            last_update = mc.get_many(last_update_keys.values())

            last_update_config_keys = {corpus: f"{corpus}:last_update_config" for corpus in corpora}
            last_update_config = mc.get_many(last_update_config_keys.values())

            # Invalidate cache for updated corpora
            for corpus in corpora:
                if last_update.get(last_update_keys[corpus], 0) < corpora[corpus]:
                    memcached_data[f"{corpus}:version"] = mc.get(f"{corpus}:version", 0) + 1
                    memcached_data[f"{corpus}:last_update"] = corpora[corpus]
                    result["corpora_invalidated"] += 1

                    # Remove outdated query data
                    for cachefile in glob.glob(os.path.join(app.config["CACHE_DIR"], "%s:*" % corpus)):
                        try:
                            if os.path.getmtime(cachefile) < corpora[corpus]:
                                os.remove(cachefile)
                                result["files_removed"] += 1
                        except FileNotFoundError:
                            pass

                if last_update_config.get(last_update_config_keys[corpus], 0) < corpora_configs.get(corpus, 0):
                    memcached_data[f"{corpus}:version_config"] = mc.get(f"{corpus}:version_config", 0) + 1
                    memcached_data[f"{corpus}:last_update_config"] = corpora_configs[corpus]
                    result["configs_invalidated"] += 1

            # If any corpus has been updated, added or removed, increase version to invalidate all combined caches
            if result["corpora_invalidated"] or not mc.get("multi:corpora", set()) == set(corpora.keys()):
                memcached_data["multi:version"] = mc.get("multi:version", 0) + 1
                memcached_data["multi:corpora"] = set(corpora.keys())
                result["multi_invalidated"] = True

            # Have any config modes or presets been updated?
            configs_updated = config_modes > mc.get("multi:config_modes", 0) or config_presets > mc.get(
                "multi:config_presets", 0)

            # If modes or presets have been updated, or any corpus config has been updated, added or removed, increase
            # version to invalidate all combined caches
            if configs_updated or result["configs_invalidated"] or not mc.get("multi:config_corpora", set()) == set(
                    corpora_configs.keys()):
                memcached_data["multi:version_config"] = mc.get("multi:version_config", 0) + 1
                memcached_data["multi:config_corpora"] = set(corpora_configs.keys())
                memcached_data["multi:config_modes"] = config_modes
                memcached_data["multi:config_presets"] = config_presets
                result["multi_config_invalidated"] = True

            mc.set_many(memcached_data)

        # Remove old query data
        for cachefile in glob.glob(os.path.join(app.config["CACHE_DIR"], "*:query_data_*")):
            try:
                if os.path.getmtime(cachefile) < (now - app.config["CACHE_LIFESPAN"] * 60):
                    os.remove(cachefile)
                    result["files_removed"] += 1
            except FileNotFoundError:
                pass
    yield result

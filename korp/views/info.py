import time

from flask import Blueprint
from pymemcache.exceptions import MemcacheError

import korp
from korp import utils
from korp.cwb import cwb
from korp.memcached import memcached

bp = Blueprint("info", __name__)


@bp.route("/sleep", methods=["GET", "POST"])
@utils.main_handler
@utils.prevent_timeout
def sleep(args):
    t = int(args.get("t", 5))
    for x in range(t):
        time.sleep(1)
        yield {"%d" % x: x}


@bp.route("/")
@bp.route("/info", methods=["GET", "POST"])
@utils.main_handler
def info(args):
    """Get version information about list of available corpora."""
    if args["cache"]:
        with memcached.get_client() as mc:
            result = mc.get("%s:info" % utils.cache_prefix(mc))
        if result:
            if "debug" in args:
                result.setdefault("DEBUG", {})
                result["DEBUG"]["cache_read"] = True
            yield result
            return

    corpora = cwb.run_cqp("show corpora;")
    version = next(corpora)

    protected = utils.get_protected_corpora()

    result = {
        "version": korp.__version__,
        "cqp_version": version,
        "corpora": list(corpora),
        "protected_corpora": protected
    }

    if args["cache"]:
        with memcached.get_client() as mc:
            added = mc.add("%s:info" % utils.cache_prefix(mc), result)
        if added and "debug" in args:
            result.setdefault("DEBUG", {})
            result["DEBUG"]["cache_saved"] = True

    yield result


@bp.route("/corpus_info", methods=["GET", "POST"])
@utils.main_handler
def corpus_info(args, no_combined_cache=False):
    """Get information about a specific corpus or corpora."""
    utils.assert_key("corpus", args, utils.IS_IDENT, True)

    corpora = utils.parse_corpora(args)

    # Check if whole query is cached
    if args["cache"]:
        checksum_combined = utils.get_hash((sorted(corpora),))
        save_cache = []
        with memcached.get_client() as mc:
            combined_cache_key = "%s:info_%s" % (utils.cache_prefix(mc), checksum_combined)
            result = mc.get(combined_cache_key)
        if result:
            if "debug" in args:
                result.setdefault("DEBUG", {})
                result["DEBUG"]["cache_read"] = True
                result["DEBUG"]["checksum"] = checksum_combined
            yield result
            return

    result = {"corpora": {}}
    total_size = 0
    total_sentences = 0

    cmd = []

    if args["cache"]:
        with memcached.get_client() as mc:
            memcached_keys = {}
            for corpus in corpora:
                memcached_keys["%s:info" % utils.cache_prefix(mc, corpus)] = corpus
            cached_corpora = mc.get_many(memcached_keys.keys())

        for key in memcached_keys:
            if key in cached_corpora:
                result["corpora"][memcached_keys[key]] = cached_corpora[key]
            else:
                save_cache.append(memcached_keys[key])

    for corpus in corpora:
        if corpus not in result["corpora"]:
            cmd += ["%s;" % corpus]
            cmd += cwb.show_attributes()
            cmd += ["info; .EOL.;"]

    if cmd:
        cmd += ["exit;"]

        # Call the CQP binary
        lines = cwb.run_cqp(cmd)

        # Skip CQP version
        next(lines)

    memcached_data = {}

    for corpus in corpora:
        if corpus in result["corpora"]:
            total_size += int(result["corpora"][corpus]["info"]["Size"])
            sentences = result["corpora"][corpus]["info"].get("Sentences", "")
            if sentences.isdigit():
                total_sentences += int(sentences)
            continue

        # Read attributes
        attrs = cwb.read_attributes(lines)

        # Corpus information
        info = {}

        for line in lines:
            if line == utils.END_OF_LINE:
                break
            if ":" in line and not line.endswith(":"):
                infokey, infoval = (x.strip() for x in line.split(":", 1))
                info[infokey] = infoval
                if infokey == "Size":
                    total_size += int(infoval)
                elif infokey == "Sentences" and infoval.isdigit():
                    total_sentences += int(infoval)

        result["corpora"][corpus] = {"attrs": attrs, "info": info}
        if args["cache"]:
            if corpus in save_cache:
                memcached_data["%s:info" % utils.cache_prefix(mc, corpus)] = result["corpora"][corpus]

    if memcached_data:
        with memcached.get_client() as mc:
            mc.set_many(memcached_data)

    result["total_size"] = total_size
    result["total_sentences"] = total_sentences

    if args["cache"] and not no_combined_cache:
        # Cache whole query
        try:
            with memcached.get_client() as mc:
                saved = mc.add(combined_cache_key, result)
        except MemcacheError:
            pass
        else:
            if saved and "debug" in args:
                result.setdefault("DEBUG", {})
                result["DEBUG"]["cache_saved"] = True
    yield result

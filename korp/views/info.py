import os
import re
import time

from flask import Blueprint
from flask import current_app as app
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
    strict = utils.parse_bool(args, "strict", False)
    if args["cache"]:
        with memcached.get_client() as mc:
            result = mc.get("%s:info_%s" % (utils.cache_prefix(mc), int(strict)))
        if result:
            if "debug" in args:
                result.setdefault("DEBUG", {})
                result["DEBUG"]["cache_read"] = True
            yield result
            return

    corpora = cwb.run_cqp("show corpora;")
    version = next(corpora)
    # CQP "show corpora" lists all corpora in the registry, but some
    # of them might nevertheless cause a "corpus undefined" error in
    # CQP, for example, because of missing data, so filter them out if
    # strict=true. However, filtering a large number of corpora slows
    # down the info command, so it is disabled by default. Caching in
    # _filter_undefined_corpora helps, though.
    if strict:
        corpora, _ = _filter_undefined_corpora(list(corpora), args["cache"])

    protected = utils.get_protected_corpora()

    result = {
        "version": korp.__version__,
        "cqp_version": version,
        "corpora": list(corpora),
        "protected_corpora": protected
    }

    if args["cache"]:
        with memcached.get_client() as mc:
            added = mc.add("%s:info_%s" % (utils.cache_prefix(mc), int(strict)), result)
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
    report_undefined_corpora = utils.parse_bool(
        args, "report_undefined_corpora", False)

    # Check if whole query is cached
    if args["cache"]:
        checksum_combined = utils.get_hash((sorted(corpora), report_undefined_corpora))
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

    if report_undefined_corpora:
        corpora, undefined_corpora = _filter_undefined_corpora(
            corpora, args["cache"], app.config["CHECK_AVAILABLE_CORPORA_STRICTLY"])

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

    if report_undefined_corpora:
        result["undefined_corpora"] = undefined_corpora

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


def _filter_undefined_corpora(corpora, caching=True, strict=True):
    """Return a pair of a list of defined and a list of undefined corpora
    in the argument corpora. If caching, check if the result is in the
    cache; if not, cache the result. If strict, try to select each
    corpus in CQP, otherwise only check the files in the CWB registry
    directory.
    """

    # Caching
    if caching:
        checksum_combined = utils.get_hash((corpora, strict))
        save_cache = []
        with memcached.get_client() as mc:
            combined_cache_key = (
                "%s:corpora_defined_%s" % (utils.cache_prefix(mc),
                                           checksum_combined))
            result = mc.get(combined_cache_key)
        if result:
            # Since this is not the result of a command, we cannot
            # add debug information on using cache to the result.
            return result
        # TODO: Add per-corpus caching

    defined = []
    undefined = []
    if strict:
        # Stricter: detects corpora that have a registry file but
        # whose data makes CQP regard them as undefined when trying to
        # use them
        cqp = [corpus.upper() + ";" for corpus in corpora]
        cqp += ["exit"]
        lines = cwb.run_cqp(cqp, errors="report")
        for line in lines:
            if line.startswith("CQP Error:"):
                matchobj = re.match(
                    r"CQP Error: Corpus ``(.+?)'' is undefined", line)
                if matchobj:
                    undefined.append(str(matchobj.group(1)))
            else:
                # SKip the rest
                break
        if undefined:
            defined = [corpus for corpus in corpora
                       if corpus not in set(undefined)]
        else:
            defined = corpora
    else:
        # It is somewhat faster but less reliable to check the
        # registry only
        registry_files = set(os.listdir(cwb.registry))
        defined = [corpus for corpus in corpora
                   if corpus.lower() in registry_files]
        undefined = [corpus for corpus in corpora
                     if corpus.lower() not in registry_files]

    result = (defined, undefined)

    if caching:
        try:
            with memcached.get_client() as mc:
                saved = mc.add(combined_cache_key, result)
        except MemcacheError:
            pass

    return result

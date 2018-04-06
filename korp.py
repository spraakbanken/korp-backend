# -*- coding: utf-8 -*-
"""
korp.py is a WSGI application for querying corpora available on the server.
Currently it acts as a wrapper for the CQP Query Language of Corpus Workbench.

Configuration is done by editing config.py.

https://spraakbanken.gu.se/korp/
"""

# Skip monkey patching if run through gunicorn (which does the patching for us)
import os
if "gunicorn" not in os.environ.get("SERVER_SOFTWARE", ""):
    from gevent import monkey
    monkey.patch_all(subprocess=False)  # Patching needs to be done as early as possible, before other imports

from gevent.pywsgi import WSGIServer
from gevent.threadpool import ThreadPool
from gevent.queue import Queue, Empty

# gunicorn patches everything, and gevent's subprocess module can't be used in
# native threads other than the main one, so we need to un-patch the subprocess module.
from importlib import reload
import subprocess
reload(subprocess)

from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dateutil.relativedelta import relativedelta
from copy import deepcopy
import datetime
import uuid
import binascii
import sys
import glob
import time
import re
import json
import zlib
import urllib.request
import urllib.parse
import urllib.error
import base64
import hashlib
import itertools
import pickle
import traceback
import functools
import math
import random
import config
from flask import Flask, request, Response, stream_with_context
from flask_mysqldb import MySQL
from flask_cors import CORS

################################################################################
# Nothing needs to be changed in this file. Use config.py for configuration.

# The version of this script
KORP_VERSION = "7.0.3"
KORP_VERSION_DATE = "2018-04-04"

# Special symbols used by this script; they must NOT be in the corpus
END_OF_LINE = "-::-EOL-::-"
LEFT_DELIM = "---:::"
RIGHT_DELIM = ":::---"

# Regular expressions for parsing parameters
IS_NUMBER = re.compile(r"^\d+$")
IS_IDENT = re.compile(r"^[\w\-,|]+$")

QUERY_DELIM = ","

################################################################################

app = Flask(__name__)
CORS(app)

# Configure database connection
app.config["MYSQL_HOST"] = config.DBHOST
app.config["MYSQL_USER"] = config.DBUSER
app.config["MYSQL_PASSWORD"] = config.DBPASSWORD
app.config["MYSQL_DB"] = config.DBNAME
app.config["MYSQL_PORT"] = config.DBPORT
app.config["MYSQL_USE_UNICODE"] = True
app.config["MYSQL_CURSORCLASS"] = "DictCursor"
mysql = MySQL(app)

# Create cache dir if needed
if config.CACHE_DIR and not os.path.exists(config.CACHE_DIR):
    os.makedirs(config.CACHE_DIR)


def main_handler(generator):
    """Decorator wrapping all WSGI endpoints, handling errors and formatting.

    Global parameters are
     - callback: an identifier that the result should be wrapped in
     - encoding: the encoding for interacting with the corpus (default: UTF-8)
     - indent: pretty-print the result with a specific indentation
     - debug: if set, return some extra information (for debugging)
    """

    @functools.wraps(generator)  # Copy original function's information, needed by Flask
    def decorated(args=None):
        internal = args is not None
        if not internal:
            if request.is_json:
                args = request.get_json()
            else:
                args = request.values.to_dict()

        if not isinstance(args.get("cache"), bool):
            args["cache"] = bool(not args.get("cache", "").lower() == "false" and
                                 config.CACHE_DIR and os.path.exists(config.CACHE_DIR))

        if internal:
            # Function is internally used
            return generator(args)
        else:
            # Function is called externally
            def error_handler():
                """Format exception info for output to user."""
                exc = sys.exc_info()
                if isinstance(exc[1], CustomTracebackException):
                    exc = exc[1].exception
                error = {"ERROR": {"type": exc[0].__name__,
                                   "value": str(exc[1])
                                   }}
                if "debug" in args:
                    error["ERROR"]["traceback"] = "".join(traceback.format_exception(*exc)).splitlines()
                return error

            def incremental_json(ff):
                """Incrementally yield result as JSON."""
                if callback:
                    yield callback + "("
                yield "{\n"

                try:
                    for response in ff:
                        if not response:
                            # Yield whitespace to prevent timeout
                            yield " \n"
                        else:
                            yield json.dumps(response)[1:-1] + ",\n"
                except GeneratorExit:
                    raise
                except:
                    error = error_handler()
                    yield json.dumps(error)[1:-1] + ",\n"

                yield json.dumps({"time": time.time() - starttime})[1:] + "\n"
                if callback:
                    yield ")"

            def full_json(ff):
                """Yield full JSON at end, but keep returning newlines to prevent timeout."""
                result = {}

                try:
                    for response in ff:
                        if not response:
                            # Yield whitespace to prevent timeout
                            yield " \n"
                        else:
                            result.update(response)
                except GeneratorExit:
                    raise
                except:
                    result = error_handler()

                result["time"] = time.time() - starttime

                if callback:
                    result = callback + "(" + json.dumps(result, indent=indent) + ")"
                else:
                    result = json.dumps(result, indent=indent)
                yield result

            starttime = time.time()
            incremental = args.get("incremental", "").lower() == "true"
            callback = args.get("callback")
            indent = int(args.get("indent", 0))

            if incremental:
                # Incremental response
                return Response(stream_with_context(incremental_json(generator(args))), mimetype="application/json")
            else:
                # We still use a streaming response even when non-incremental, to prevent timeouts
                return Response(stream_with_context(full_json(generator(args))), mimetype="application/json")

    return decorated


################################################################################
# ARGUMENT PARSING
################################################################################

def parse_corpora(args):
    corpora = args.get("corpus")
    if isinstance(corpora, str):
        corpora = corpora.upper().split(QUERY_DELIM)
    return sorted(set(corpora))


def parse_within(args):
    within = defaultdict(lambda: args.get("defaultwithin"))

    if args.get("within"):
        if ":" not in args.get("within"):
            raise ValueError("Malformed value for key 'within'.")
        within.update(dict(x.split(":") for x in args.get("within").split(QUERY_DELIM)))
    return within


def parse_cqp_subcqp(args):
    cqp = [args.get(key) for key in sorted([k for k in args.keys() if k.startswith("cqp")],
                                           key=lambda x: int(x[3:]) if len(x) > 3 else 0)]
    subcqp = [args.get(key) for key in sorted([k for k in args.keys() if k.startswith("subcqp")],
                                              key=lambda x: int(x[6:]) if len(x) > 6 else 0)]
    return cqp, subcqp


################################################################################
# INFO
################################################################################

@app.route("/sleep", methods=["GET", "POST"])
@main_handler
def sleep(args):
    t = int(args.get("t", 5))
    for x in range(t):
        time.sleep(1)
        yield {"%d" % x: x}


@app.route("/", methods=["GET", "POST"])
@app.route("/info", methods=["GET", "POST"])
@main_handler
def info(args):
    """Return information, either about a specific corpus
    or general information about the available corpora.
    """
    if args.get("corpus"):
        yield corpus_info(args)
    else:
        yield general_info(args)


def general_info(args):
    """Return information about the available corpora.
    """

    if args["cache"]:
        cache_filename = os.path.join(config.CACHE_DIR, "info")
        if os.path.isfile(cache_filename):
            with open(cache_filename, "r") as cachefile:
                result = json.load(cachefile)
            if "debug" in args:
                result.setdefault("DEBUG", {})
                result["DEBUG"]["cache_read"] = True
            return result

    corpora = run_cqp("show corpora;")
    version = next(corpora)
    protected = []

    if config.PROTECTED_FILE:
        with open(config.PROTECTED_FILE) as infile:
            protected = [x.strip() for x in infile.readlines()]

    result = {"version": KORP_VERSION, "cqp-version": version, "corpora": list(corpora), "protected_corpora": protected}

    if args["cache"]:
        if not os.path.exists(cache_filename):
            cache_filename_temp = "%s.%s" % (cache_filename, str(uuid.uuid4()))

            with open(cache_filename_temp, "w") as cachefile:
                json.dump(result, cachefile)
            os.rename(cache_filename_temp, cache_filename)

            if "debug" in args:
                result.setdefault("DEBUG", {})
                result["DEBUG"]["cache_saved"] = True

    return result


def corpus_info(args, no_combined_cache=False):
    """Return information about a specific corpus or corpora.
    """
    assert_key("corpus", args, IS_IDENT, True)

    corpora = parse_corpora(args)

    # Check if whole query is cached
    if args["cache"]:
        checksum_combined = get_hash((sorted(corpora),))
        save_cache = []
        cache_filename = os.path.join(config.CACHE_DIR, "info_" + checksum_combined)
        if os.path.exists(cache_filename):
            with open(cache_filename, "r") as f:
                result = json.load(f)
            if "debug" in args:
                result.setdefault("DEBUG", {})
                result["DEBUG"]["cache_read"] = True
                result["DEBUG"]["checksum"] = checksum_combined
            return result

    result = {"corpora": {}}
    total_size = 0
    total_sentences = 0

    cmd = []

    for corpus in corpora:
        # Check if corpus is cached
        if args["cache"]:
            cache_filename = os.path.join(config.CACHE_DIR, "%s:info" % corpus)
            if os.path.exists(cache_filename):
                with open(cache_filename, "r") as f:
                    result["corpora"][corpus] = json.load(f)
            else:
                save_cache.append(corpus)
        if corpus not in result["corpora"]:
            cmd += ["%s;" % corpus]
            cmd += show_attributes()
            cmd += ["info; .EOL.;"]

    if cmd:
        cmd += ["exit;"]

        # Call the CQP binary
        lines = run_cqp(cmd)

        # Skip CQP version
        next(lines)

    for corpus in corpora:
        if corpus in result["corpora"]:
            total_size += int(result["corpora"][corpus]["info"]["Size"])
            total_sentences += int(result["corpora"][corpus]["info"].get("Sentences", 0))
            continue

        # Read attributes
        attrs = read_attributes(lines)

        # Corpus information
        info = {}

        for line in lines:
            if line == END_OF_LINE:
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
                cache_filename = os.path.join(config.CACHE_DIR, "%s:info" % corpus)
                cache_filename_temp = "%s.%s" % (cache_filename, str(uuid.uuid4()))
                with open(cache_filename_temp, "w") as f:
                    json.dump(result["corpora"][corpus], f)
                os.rename(cache_filename_temp, cache_filename)

    result["total_size"] = total_size
    result["total_sentences"] = total_sentences

    if args["cache"] and not no_combined_cache:
        # Cache whole query
        cache_filename = os.path.join(config.CACHE_DIR, "info_" + checksum_combined)
        if not os.path.exists(cache_filename):
            cache_filename_temp = "%s.%s" % (cache_filename, str(uuid.uuid4()))

            with open(cache_filename_temp, "w") as f:
                json.dump(result, f)
            os.rename(cache_filename_temp, cache_filename)

            if "debug" in args:
                result.setdefault("DEBUG", {})
                result["DEBUG"]["cache_saved"] = True

    return result


################################################################################
# QUERY
################################################################################

@app.route("/query_sample", methods=["GET", "POST"])
@main_handler
def query_sample(args):
    """Run a sequential query in the selected corpora in random order until at least one
    hit is found, and then abort the query. Use to get a random sample sentence."""

    corpora = parse_corpora(args)
    # Randomize corpus order
    random.shuffle(corpora)

    for i in range(len(corpora)):
        corpus = corpora[i]
        check_authentication([corpus])

        args["corpus"] = corpus
        args["sort"] = "random"

        result = generator_to_dict(query(args))
        if result["hits"] > 0:
            yield result
            return

    yield result


@app.route("/query", methods=["GET", "POST"])
@main_handler
def query(args):
    """Perform a CQP query and return a number of matches.

    Each match contains position information and a list of the words and attributes in the match.

    The required parameters are
     - corpus: the CWB corpus. More than one parameter can be used.
     - cqp: the CQP query string
     - start, end: which result rows that should be returned

    The optional parameters are
     - context: how many words/sentences to the left/right should be returned
       (default '10 words')
     - show: add once for each corpus parameter (positional/strutural/alignment)
       (default only show the 'word' parameter)
     - show_struct: structural annotations for matched region. Multiple parameters possible.
     - within: only search for matches within the given s-attribute (e.g., within a sentence)
       (default: no within)
     - cut: set cutoff threshold to reduce the size of the result
       (default: no cutoff)
     - sort: sort the results by keyword ('keyword'), left or right context ('left'/'right') or random ('random')
       (default: no sorting)
     - incremental: returns the result incrementally instead of all at once
    """
    assert_key("cqp", args, r"", True)
    assert_key("corpus", args, IS_IDENT, True)
    assert_key("start", args, IS_NUMBER)
    assert_key("end", args, IS_NUMBER)
    # assert_key("context", args, r"^\d+ [\w-]+$")
    assert_key("show", args, IS_IDENT)
    assert_key("show_struct", args, IS_IDENT)
    # assert_key("within", args, IS_IDENT)
    assert_key("cut", args, IS_NUMBER)
    assert_key("sort", args, r"")
    assert_key("incremental", args, r"(true|false)")

    incremental = args.get("incremental", "").lower() == "true"
    free_search = args.get("in_order", "").lower() == "false"
    use_cache = args["cache"]
    cut = args.get("cut")

    corpora = parse_corpora(args)

    check_authentication(corpora)

    show = args.get("show") or []  # We don't use .get("show", []) since "show" might be the empty string.
    if isinstance(show, str):
        show = show.split(QUERY_DELIM)
    show = set(show + ["word"])

    show_structs = args.get("show_struct") or []
    if isinstance(show_structs, str):
        show_structs = show_structs.split(QUERY_DELIM)
    show_structs = set(show_structs)

    expand_prequeries = not args.get("expand_prequeries", "").lower() == "false"

    start, end = int(args.get("start") or 0), int(args.get("end") or 9)

    if config.MAX_KWIC_ROWS and end - start >= config.MAX_KWIC_ROWS:
        raise ValueError("At most %d KWIC rows can be returned per call." % config.MAX_KWIC_ROWS)

    within = parse_within(args)

    # Parse "context"/"leftcontext"/"rightcontext"/"defaultcontext"
    defaultcontext = args.get("defaultcontext") or "10 words"
    context = defaultdict(lambda: (defaultcontext,))
    contexts = {}

    for c in ("leftcontext", "rightcontext", "context"):
        cv = args.get(c, "")
        if cv:
            if ":" not in cv:
                raise ValueError("Malformed value for key '%s'." % c)
            contexts[c] = dict(x.split(":") for x in cv.split(QUERY_DELIM))
        else:
            contexts[c] = {}

    for corpus in set(k for v in contexts.values() for k in v.keys()):
        if corpus in contexts["leftcontext"] or corpus in contexts["rightcontext"]:
            context[corpus] = (contexts["leftcontext"].get(corpus, defaultcontext),
                               contexts["rightcontext"].get(corpus, defaultcontext))
        else:
            context[corpus] = (contexts["context"].get(corpus, defaultcontext),)

    # Sort numbered CQP-queries numerically
    cqp, _ = parse_cqp_subcqp(args)

    # Parameters used for all queries
    queryparams = {"free_search": free_search,
                   "use_cache": use_cache,
                   "show": show,
                   "show_structs": show_structs,
                   "expand_prequeries": expand_prequeries,
                   "cut": cut,
                   "cqp": cqp
                   }

    result = {"kwic": []}

    # Checksum for whole query, used to verify 'querydata' from the client
    checksum = get_hash((sorted(corpora),
                         cqp,
                         sorted(within.items()),
                         cut,
                         expand_prequeries,
                         free_search))

    debug = {}
    if "debug" in args:
        debug["checksum"] = checksum

    ns = Namespace()
    ns.total_hits = 0
    statistics = {}

    saved_statistics = {}
    querydata = args.get("querydata")

    if querydata:
        try:
            querydata = zlib.decompress(base64.b64decode(
                querydata.replace("\\n", "\n").replace("-", "+").replace("_", "/"))).decode("UTF-8")
        except:
            if "debug" in args:
                debug["querydata_unparseable"] = True
        else:
            if "debug" in args:
                debug["querydata_read"] = True
            saved_checksum, stats_temp = querydata.split(";", 1)
            if saved_checksum == checksum:
                for pair in stats_temp.split(";"):
                    corpus, hits = pair.split(":")
                    saved_statistics[corpus] = int(hits)
            elif "debug" in args:
                debug["querydata_checksum_mismatch"] = True

    if use_cache and not saved_statistics:
        # Querydata parsing failed or was missing, so look for cached hits instead
        for corpus in corpora:
            corpus_checksum = get_hash((cqp,
                                        within[corpus],
                                        cut,
                                        expand_prequeries,
                                        free_search))

            cache_hits_filename = os.path.join(config.CACHE_DIR, "%s:query_size_%s" % (corpus, corpus_checksum))
            if os.path.isfile(cache_hits_filename):
                with open(cache_hits_filename, "r") as cache_hits:
                    saved_statistics[corpus] = int(cache_hits.read())

    ns.start_local = start
    ns.end_local = end

    if saved_statistics:
        if "debug" in args:
            debug["cache_coverage"] = "%d/%d" % (len(saved_statistics), len(corpora))
        complete_hits = set(corpora) == set(saved_statistics.keys())
    else:
        complete_hits = False

    if complete_hits:
        # We have saved_statistics available for all corpora, so calculate which
        # corpora need to be queried and then query them in parallel.
        corpora_hits = which_hits(corpora, saved_statistics, start, end)
        ns.total_hits = sum(saved_statistics.values())
        statistics = saved_statistics
        corpora_kwics = {}
        ns.progress_count = 0

        if len(corpora_hits) == 0:
            pass
        elif len(corpora_hits) == 1:
            # If only hits in one corpus, it is faster to not use threads
            corpus, hits = list(corpora_hits.items())[0]

            def _query_single_corpus(queue):
                result["kwic"], _ = query_and_parse(corpus, within=within[corpus], context=context[corpus],
                                                    start=hits[0], end=hits[1], **queryparams)
                queue.put("DONE")

            for msg in prevent_timeout(_query_single_corpus):
                yield msg
        else:
            if incremental:
                yield {"progress_corpora": list(corpora_hits.keys())}

            def _query_corpora_in_parallel(queue):
                with ThreadPoolExecutor(max_workers=config.PARALLEL_THREADS) as executor:
                    future_query = dict(
                        (executor.submit(query_and_parse, corpus, within=within[corpus], context=context[corpus],
                                         start=corpora_hits[corpus][0], end=corpora_hits[corpus][1], **queryparams),
                         corpus)
                        for corpus in corpora_hits)

                    for future in futures.as_completed(future_query):
                        corpus = future_query[future]
                        if future.exception() is not None:
                            raise CQPError(future.exception())
                        else:
                            kwic, _ = future.result()
                            corpora_kwics[corpus] = kwic
                            if incremental:
                                queue.put({"progress_%d" % ns.progress_count: {"corpus": corpus,
                                                                               "hits": corpora_hits[corpus][1] -
                                                                                       corpora_hits[corpus][0] + 1}})
                                ns.progress_count += 1
                    queue.put("DONE")

            for msg in prevent_timeout(_query_corpora_in_parallel):
                yield msg

            for corpus in corpora:
                if corpus in corpora_hits.keys():
                    result["kwic"].extend(corpora_kwics[corpus])
    else:
        # saved_statistics is missing or incomplete, so we need to query the corpora in
        # serial until we have the needed rows, and then query the remaining corpora
        # in parallel to get number of hits.
        if incremental:
            yield {"progress_corpora": corpora}
        ns.progress_count = 0
        ns.rest_corpora = []

        def _query_corpora_in_serial(queue):
            # Serial until we've got all the requested rows
            for i, corpus in enumerate(corpora):
                if ns.end_local < 0:
                    ns.rest_corpora = corpora[i:]
                    break
                skip_corpus = False
                if corpus in saved_statistics:
                    nr_hits = saved_statistics[corpus]
                    if nr_hits - 1 < ns.start_local:
                        kwic = []
                        skip_corpus = True

                if not skip_corpus:
                    kwic, nr_hits = query_and_parse(corpus, within=within[corpus], context=context[corpus],
                                                    start=ns.start_local, end=ns.end_local, **queryparams)

                statistics[corpus] = nr_hits
                ns.total_hits += nr_hits

                # Calculate which hits from next corpus we need, if any
                ns.start_local -= nr_hits
                ns.end_local -= nr_hits
                if ns.start_local < 0:
                    ns.start_local = 0

                result["kwic"].extend(kwic)

                if incremental:
                    queue.put({"progress_%d" % ns.progress_count: {"corpus": corpus, "hits": nr_hits}})
                    ns.progress_count += 1

            queue.put("DONE")

        for msg in prevent_timeout(_query_corpora_in_serial):
            yield msg

        if incremental:
            yield result
            result = {}

        if ns.rest_corpora:
            if saved_statistics:
                for corpus in ns.rest_corpora:
                    if corpus in saved_statistics:
                        statistics[corpus] = saved_statistics[corpus]
                        ns.total_hits += saved_statistics[corpus]

            def _get_total_in_parallel(queue):
                with ThreadPoolExecutor(max_workers=config.PARALLEL_THREADS) as executor:
                    future_query = dict(
                        (executor.submit(query_corpus, corpus, within=within[corpus],
                                         context=context[corpus], start=0, end=0, no_results=True, **queryparams),
                         corpus)
                        for corpus in ns.rest_corpora if corpus not in saved_statistics)

                    for future in futures.as_completed(future_query):
                        corpus = future_query[future]
                        if future.exception() is not None:
                            raise CQPError(future.exception())
                        else:
                            _, nr_hits, _ = future.result()
                            statistics[corpus] = nr_hits
                            ns.total_hits += nr_hits
                            if incremental:
                                queue.put({"progress_%d" % ns.progress_count: {"corpus": corpus, "hits": nr_hits}})
                                ns.progress_count += 1
                queue.put("DONE")

            for msg in prevent_timeout(_get_total_in_parallel):
                yield msg

    if "debug" in args:
        debug["cqp"] = cqp

    result["hits"] = ns.total_hits
    result["corpus_hits"] = statistics
    result["corpus_order"] = corpora
    result["querydata"] = binascii.b2a_base64(zlib.compress(
        bytes(checksum + ";" + ";".join("%s:%d" % (c, h) for c, h in statistics.items()),
              "utf-8"))).decode("utf-8").replace("+", "-").replace("/", "_")

    if debug:
        result["DEBUG"] = debug

    # Log query
    if config.LOG_FILE2:
        try:
            ip = hashlib.md5(request.environ.get("HTTP_X_FORWARDED_FOR", "N/A").encode("UTF-8")).hexdigest()
            with open(config.LOG_FILE2, "a") as o:
                o.write("%s\t%s\t%s\t%s\t%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ip,
                        "".join(cqp).replace("\n", " "), start, ",".join(corpora).upper()))
        except:
            pass

    yield result


@app.route("/optimize", methods=["GET", "POST"])
@main_handler
def optimize(args):
    assert_key("cqp", args, r"", True)

    cqpparams = {"within": args.get("within") or "sentence"}
    if args.get("cut"):
        cqpparams["cut"] = args["cut"]

    free_search = args.get("in_order", "").lower() == "false"

    cqp = args["cqp"]
    result = {"cqp": query_optimize(cqp, cqpparams, find_match=False, expand=False, free_search=free_search)}
    yield result


def query_optimize(cqp, cqpparams, find_match=True, expand=True, free_search=False):
    """ Optimize simple queries with multiple words by converting them to MU queries.
        Optimization only works for queries with at least two tokens, or one token preceded
        by one or more wildcards. The query also must use "within".
        Return a tuple (return code, query)
        0 = optimization successful
        1 = optimization not needed (e.g. single word searches)
        2 = optimization not possible (e.g. searches with repetition of non-wildcards)
        """
    # Split query into tokens
    tokens, rest = parse_cqp(cqp)
    within = cqpparams.get("within")

    leading_wildcards = False

    # Don't allow wildcards in free searches
    if free_search:
        if any([token.startswith("[]") for token in tokens]):
            raise CQPError("Wildcards not allowed in free order query.")
    else:
        # Remove leading and trailing wildcards since they will only slow us down
        while tokens and tokens[0].startswith("[]"):
            leading_wildcards = True
            del tokens[0]
        while tokens and tokens[-1].startswith("[]"):
            del tokens[-1]

    if len(tokens) == 0 or (len(tokens) == 1 and not leading_wildcards):
        # Query doesn't benefit from optimization
        return 1, make_query(make_cqp(cqp, **cqpparams))
    elif rest or not within:
        # Couldn't optimize this query
        return 2, make_query(make_cqp(cqp, **cqpparams))

    cmd = ["MU"]
    wildcards = {}

    for i in range(len(tokens) - 1):
        if tokens[i].startswith("[]"):
            n1 = n2 = None
            if tokens[i] == "[]":
                n1 = n2 = 1
            elif re.search(r"{\s*(\d+)\s*,\s*(\d*)\s*}$", tokens[i]):
                n = re.search(r"{\s*(\d+)\s*,\s*(\d*)\s*}$", tokens[i]).groups()
                n1 = int(n[0])
                n2 = int(n[1]) if n[1] else 9999
            elif re.search(r"{\s*(\d*)\s*}$", tokens[i]):
                n1 = n2 = int(re.search(r"{\s*(\d*)\s*}$", tokens[i]).groups()[0])
            if n1 is not None:
                wildcards[i] = (n1, n2)
            continue
        elif re.search(r"{.*?}$", tokens[i]):
            # Repetition for anything other than wildcards can't be optimized
            return 2, make_query(make_cqp(cqp, **cqpparams))
        cmd[0] += " (meet %s" % (tokens[i])

    if re.search(r"{.*?}$", tokens[-1]):
        # Repetition for anything other than wildcards can't be optimized
        return 2, make_query(make_cqp(cqp, **cqpparams))

    cmd[0] += " %s" % tokens[-1]

    wildcard_range = [1, 1]
    for i in range(len(tokens) - 2, -1, -1):
        if i in wildcards:
            wildcard_range[0] += wildcards[i][0]
            wildcard_range[1] += wildcards[i][1]
            continue
        elif i + 1 in wildcards:
            if wildcard_range[1] >= 9999:
                cmd[0] += " %s)" % within
            else:
                cmd[0] += " %d %d)" % (wildcard_range[0], wildcard_range[1])
            wildcard_range = [1, 1]
        elif free_search:
            cmd[0] += " %s)" % within
        else:
            cmd[0] += " 1 1)"

    if find_match and not free_search:
        # MU searches only highlight the first keyword of each hit. To highlight all keywords we need to
        # do a new non-optimized search within the results, and to be able to do that we first need to expand the rows.
        # Most of the times we only need to expand to the right, except for when leading wildcards are used.
        if leading_wildcards:
            cmd[0] += " expand to %s;" % within
        else:
            cmd[0] += " expand right to %s;" % within
        cmd += ["Last;"]
        cmd += make_query(make_cqp(cqp, **cqpparams))
    elif expand or free_search:
        cmd[0] += " expand to %s;" % within
    else:
        cmd[0] += ";"

    return 0, cmd


def query_corpus(corpus, cqp, within=None, cut=None, context=None, show=None, show_structs=None, start=0, end=10,
                 sort=None, random_seed=None,
                 no_results=False, expand_prequeries=True, free_search=False, use_cache=False):
    if use_cache:
        # Calculate checksum
        # Needs to contain all arguments that may influence the results
        checksum_data = (cqp,
                         within,
                         cut,
                         expand_prequeries,
                         free_search)

        checksum = get_hash(checksum_data)
        unique_id = str(uuid.uuid4())

        cache_query = "query_data_%s" % checksum
        cache_query_temp = cache_query + "_" + unique_id

        cache_filename = os.path.join(config.CACHE_DIR, "%s:query_data_%s" % (corpus, checksum))
        cache_filename_temp = cache_filename + "_" + unique_id

        cache_size_filename = os.path.join(config.CACHE_DIR, "%s:query_size_%s" % (corpus, checksum))
        cache_size_filename_temp = cache_size_filename + "." + unique_id

        is_cached = os.path.isfile(cache_filename) and os.path.isfile(cache_size_filename)
        if is_cached:
            with open(cache_size_filename, "r") as f:
                cache_hits = f.read()
        cached_no_hits = is_cached and int(cache_hits) == 0
    else:
        is_cached = False

    # Optimization
    do_optimize = True

    show = show.copy()  # To not edit the original

    cqpparams = {"within": within,
                 "cut": cut}

    # Handle aligned corpora
    if "|" in corpus:
        linked = corpus.split("|")
        cqpnew = []

        for c in cqp:
            cs = c.split("LINKED_CORPUS:")

            # In a multi-language query, the "within" argument must be placed directly
            # after the main (first language) query
            if len(cs) > 1 and within:
                cs[0] = "%s within %s : " % (cs[0].rstrip()[:-1], within)
                del cqpparams["within"]

            c = [cs[0]]

            for d in cs[1:]:
                linked_corpora, link_cqp = d.split(None, 1)
                if linked[1] in linked_corpora.split("|"):
                    c.append("%s %s" % (linked[1], link_cqp))

            cqpnew.append("".join(c).rstrip(": "))

        cqp = cqpnew
        corpus = linked[0]
        show.add(linked[1].lower())

    # Sorting
    if sort == "left":
        sortcmd = ["sort by word on match[-1] .. match[-3];"]
    elif sort == "keyword":
        sortcmd = ["sort by word;"]
    elif sort == "right":
        sortcmd = ["sort by word on matchend[1] .. matchend[3];"]
    elif sort == "random":
        sortcmd = ["sort randomize %s;" % (random_seed or "")]
    elif sort:
        # Sort by positional attribute
        sortcmd = ["sort by %s;" % sort]
    else:
        sortcmd = []

    # Build the CQP query
    cmd = []

    if use_cache:
        cmd += ['set DataDirectory "%s";' % config.CACHE_DIR]

    cmd += ["%s;" % corpus]

    # This prints the attributes and their relative order:
    cmd += show_attributes()

    retcode = 0

    if is_cached:
        # This exact query has been done before. Read corpus positions from cache.
        if not cached_no_hits:
            cmd += ["Last = %s;" % cache_query]
    else:
        for i, c in enumerate(cqp):
            cqpparams_temp = cqpparams.copy()
            pre_query = i + 1 < len(cqp)

            if pre_query and expand_prequeries:
                cqpparams_temp["expand"] = "to " + within

            if free_search:
                retcode, free_query = query_optimize(c, cqpparams_temp, free_search=True)
                if retcode == 2:
                    raise CQPError("Couldn't convert into free order query.")
                cmd += free_query
            elif do_optimize and expand_prequeries:
                # If expand_prequeries is False, we can't use optimization
                cmd += query_optimize(c, cqpparams_temp, find_match=(not pre_query))[1]
            else:
                cmd += make_query(make_cqp(c, **cqpparams_temp))

            if pre_query:
                cmd += ["Last;"]

    if use_cache and cached_no_hits:
        # Print EOL if no hits
        cmd += [".EOL.;"]
    else:
        # This prints the size of the query (i.e., the number of results):
        cmd += ["size Last;"]

    if use_cache and not is_cached:
        cmd += ["%s = Last; save %s;" % (cache_query_temp, cache_query_temp)]

    if not no_results and not (use_cache and cached_no_hits):
        if free_search and retcode == 0:
            tokens, _ = parse_cqp(cqp[-1])
            cmd += ["Last;"]
            cmd += ["cut %s %s;" % (start, end)]
            cmd += make_query(make_cqp("(%s)" % " | ".join(set(tokens)), **cqpparams))

        cmd += ["show +%s;" % " +".join(show)]
        if len(context) == 1:
            cmd += ["set Context %s;" % context[0]]
        else:
            cmd += ["set LeftContext %s;" % context[0]]
            cmd += ["set RightContext %s;" % context[1]]
        cmd += ["set LeftKWICDelim '%s '; set RightKWICDelim ' %s';" % (LEFT_DELIM, RIGHT_DELIM)]
        if show_structs:
            cmd += ["set PrintStructures '%s';" % ", ".join(show_structs)]
        cmd += ["set ExternalSort yes;"]
        cmd += sortcmd
        if free_search:
            cmd += ["cat Last;"]
        else:
            cmd += ["cat Last %s %s;" % (start, end)]

    cmd += ["exit;"]

    ######################################################################
    # Then we call the CQP binary, and read the results

    lines = run_cqp(cmd, attr_ignore=True)

    # Skip the CQP version
    next(lines)

    # Read the attributes and their relative order
    attrs = read_attributes(lines)

    # Read the size of the query, i.e., the number of results
    nr_hits = next(lines)
    nr_hits = 0 if nr_hits == END_OF_LINE else int(nr_hits)

    if use_cache and not is_cached:
        # Save number of hits
        with open(cache_size_filename_temp, "w") as f:
            f.write("%d\n" % nr_hits)

        os.rename(cache_size_filename_temp, cache_size_filename)

        try:
            os.rename(cache_filename_temp, cache_filename)
        except FileNotFoundError:
            pass

    return lines, nr_hits, attrs


def query_parse_lines(corpus, lines, attrs, show, show_structs, free_matches=False):
    """Parse concordance lines from CWB."""

    # Filter out unavailable attributes
    p_attrs = [attr for attr in attrs["p"] if attr in show]
    nr_splits = len(p_attrs) - 1
    s_attrs = set(attr for attr in attrs["s"] if attr in show)
    ls_attrs = set(attr for attr in attrs["s"] if attr in show_structs)
    # a_attrs = set(attr for attr in attrs["a"] if attr in shown)

    last_line_span = ()

    kwic = []
    for line in lines:
        linestructs = {}
        match = {}

        header, line = line.split(":", 1)
        if header[:3] == "-->":
            # For aligned corpora, every other line is the aligned result
            aligned = header[3:]
        else:
            # This is the result row for the query corpus
            aligned = None
            match["position"] = int(header)

        # Handle PrintStructures
        if ls_attrs and not aligned:
            if ":  " in line:
                lineattr, line = line.rsplit(":  ", 1)
            else:
                # Sometimes, depending on context, CWB uses only one space instead of two as a separator
                lineattr, line = line.split(">: ", 1)
                lineattr += ">"

            lineattrs = lineattr[2:-1].split("><")

            # Handle "><" in attribute values
            if not len(lineattrs) == len(ls_attrs):
                new_lineattrs = []
                for la in lineattrs:
                    if not la.split(" ", 1)[0] in ls_attrs:
                        new_lineattrs[-1] += "><" + la
                    else:
                        new_lineattrs.append(la)
                lineattrs = new_lineattrs

            for s in lineattrs:
                if s in ls_attrs:
                    s_key = s
                    s_val = None
                else:
                    s_key, s_val = s.split(" ", 1)

                linestructs[s_key] = s_val

        words = line.split()
        tokens = []
        n = 0
        structs = defaultdict(list)
        struct = None
        struct_value = []

        try:
            for word in words:
                if struct:
                    # Structural attrs can be split in the middle (<s_n 123>),
                    # so we need to finish the structure here
                    if ">" not in word:
                        struct_value.append(word)
                        continue

                    struct_v, word = word.split(">", 1)
                    structs["open"].append(struct + " " + " ".join(struct_value + [struct_v]))
                    struct = None
                    struct_value = []

                # We use special delimiters to see when we enter and leave the match region
                if word == LEFT_DELIM:
                    match["start"] = n
                    continue
                elif word == RIGHT_DELIM:
                    match["end"] = n
                    continue

                # We read all structural attributes that are opening (from the left)
                while word[0] == "<":
                    if word[1:] in s_attrs:
                        # We have found a structural attribute with a value (<s_n 123>).
                        # We continue to the next word to get the value
                        struct = word[1:]
                        break
                    elif ">" in word and word[1:word.find(">")] in s_attrs:
                        # We have found a structural attribute without a value (<s>)
                        struct, word = word[1:].split(">", 1)
                        structs["open"].append(struct)
                        struct = None
                    else:
                        # What we've found is not a structural attribute
                        break

                if struct:
                    # If we stopped in the middle of a struct (<s_n 123>),
                    # we need to continue with the next word
                    continue

                # Now we read all s-attrs that are closing (from the right)
                while word[-1] == ">" and "</" in word:
                    tempword, struct = word[:-1].rsplit("</", 1)
                    if not tempword or struct not in s_attrs:
                        struct = None
                        break
                    elif struct in s_attrs:
                        word = tempword
                        structs["close"].insert(0, struct)
                        struct = None

                # What's left is the word with its p-attrs
                values = word.rsplit("/", nr_splits)
                token = dict((attr, translate_undef(val)) for (attr, val) in zip(p_attrs, values))
                if structs:
                    token["structs"] = structs
                    structs = defaultdict(list)
                tokens.append(token)

                n += 1
        except IndexError:
            # Attributes containing ">" or "<" can make some lines unparseable. We skip them
            # until we come up with better a solution.
            continue

        if aligned:
            # If this was an aligned row, we add it to the previous kwic row
            if words != ["(no", "alignment", "found)"]:
                kwic[-1].setdefault("aligned", {})[aligned] = tokens
        else:
            if "start" not in match:
                # TODO: CQP bug - CQP can't handle too long sentences, skipping
                continue
            # Otherwise we add a new kwic row
            kwic_row = {"corpus": corpus, "match": match if not free_matches else [match]}
            if linestructs:
                kwic_row["structs"] = linestructs
            kwic_row["tokens"] = tokens

            if free_matches:
                line_span = (match["position"] - match["start"], match["position"] - match["start"] + len(tokens) - 1)
                if line_span == last_line_span:
                    kwic[-1]["match"].append(match)
                else:
                    kwic.append(kwic_row)
                last_line_span = line_span
            else:
                kwic.append(kwic_row)

    return kwic


def query_and_parse(corpus, cqp, within=None, cut=None, context=None, show=None, show_structs=None, start=0, end=10,
                    sort=None, random_seed=None, no_results=False, expand_prequeries=True, free_search=False,
                    use_cache=False):
    lines, nr_hits, attrs = query_corpus(corpus, cqp, within, cut, context, show, show_structs, start, end, sort,
                                         random_seed, no_results, expand_prequeries, free_search, use_cache)
    kwic = query_parse_lines(corpus, lines, attrs, show, show_structs, free_matches=free_search)
    return kwic, nr_hits


def which_hits(corpora, stats, start, end):
    corpus_hits = {}
    for corpus in corpora:
        hits = stats[corpus]
        if hits > start:
            corpus_hits[corpus] = (start, min(hits - 1, end))

        start -= hits
        end -= hits
        if start < 0:
            start = 0
        if end < 0:
            break

    return corpus_hits


@app.route("/struct_values", methods=["GET", "POST"])
@main_handler
def struct_values(args):
    assert_key("corpus", args, IS_IDENT, True)
    assert_key("struct", args, re.compile(r"^[\w_\d,>]+$"), True)
    assert_key("incremental", args, r"(true|false)")

    incremental = args.get("incremental", "").lower() == "true"

    stats = args.get("count", "").lower() == "true"

    corpora = parse_corpora(args)

    check_authentication(corpora)

    structs = args.get("struct")
    if isinstance(structs, str):
        structs = structs.split(QUERY_DELIM)

    split = args.get("split", "")
    if isinstance(split, str):
        split = split.split(QUERY_DELIM)

    ns = Namespace()  # To make variables writable from nested functions

    result = {"corpora": defaultdict(dict)}
    total_stats = defaultdict(set)

    from_cache = set()  # Keep track of what has been read from cache

    if args["cache"]:
        all_cache = True
        for corpus in corpora:
            for struct in structs:
                checksum_data = (corpus, struct, split, stats)
                checksum = get_hash(checksum_data)

                cache_filename = os.path.join(config.CACHE_DIR, "%s:struct_values_%s" % (corpus, checksum))
                if os.path.exists(cache_filename):
                    with open(cache_filename, "r") as f:
                        data = json.load(f)
                    result["corpora"].setdefault(corpus, {})
                    if data:
                        result["corpora"][corpus][struct] = data
                    if "debug" in args:
                        result.setdefault("DEBUG", {"caches_read": []})
                        result["DEBUG"]["caches_read"].append("%s:%s" % (corpus, struct))
                    from_cache.add((corpus, struct))
                else:
                    all_cache = False

        if all_cache:
            result["combined"] = {}
            yield result
            return

    ns.progress_count = 0
    if incremental:
        yield ({"progress_corpora": list(corpora)})

    def anti_timeout(queue):
        with ThreadPoolExecutor(max_workers=config.PARALLEL_THREADS) as executor:
            future_query = dict((executor.submit(count_query_worker_simple, corpus, cqp=None,
                                                 groupby=[(s, True) for s in struct.split(">")],
                                                 use_cache=args["cache"]), (corpus, struct))
                                for corpus in corpora for struct in structs if not (corpus, struct) in from_cache)

            for future in futures.as_completed(future_query):
                corpus, struct = future_query[future]
                if future.exception() is not None:
                    raise CQPError(future.exception())
                else:
                    lines, nr_hits, corpus_size = future.result()

                    corpus_stats = {} if stats else set()
                    vals_dict = {}
                    struct_list = struct.split(">")

                    for line in lines:
                        freq, val = line.lstrip().split(" ", 1)

                        if ">" in struct:
                            vals = val.split("\t")

                            if split:
                                vals = [[x for x in n.split("|") if x] if struct_list[i] in split and n else [n] for
                                        i, n in enumerate(vals)]
                                vals_prod = itertools.product(*vals)
                            else:
                                vals_prod = [vals]

                            for val in vals_prod:
                                prev = vals_dict
                                for i, n in enumerate(val):
                                    if stats and i == len(val) - 1:
                                        prev.setdefault(n, 0)
                                        prev[n] += int(freq)
                                        break
                                    elif not stats and i == len(val) - 1:
                                        prev.append(n)
                                        break
                                    elif not stats and i == len(val) - 2:
                                        prev.setdefault(n, [])
                                    else:
                                        prev.setdefault(n, {})
                                    prev = prev[n]
                        else:
                            if struct in split:
                                vals = [x for x in val.split("|") if x] if val else [""]
                            else:
                                vals = [val]
                            for val in vals:
                                if stats:
                                    corpus_stats[val] = int(freq)
                                else:
                                    corpus_stats.add(val)

                    if ">" in struct:
                        result["corpora"][corpus][struct] = vals_dict
                    elif corpus_stats:
                        result["corpora"][corpus][struct] = corpus_stats if stats else sorted(corpus_stats)

                    if incremental:
                        queue.put({"progress_%d" % ns.progress_count: corpus})
                        ns.progress_count += 1
            queue.put("DONE")

    for msg in prevent_timeout(anti_timeout):
        yield msg

    result["combined"] = {}

    if args["cache"]:
        unique_id = str(uuid.uuid4())

        for corpus in corpora:
            for struct in structs:
                if (corpus, struct) in from_cache:
                    continue
                checksum_data = (corpus, struct, split, stats)
                checksum = get_hash(checksum_data)
                cache_filename = os.path.join(config.CACHE_DIR, "%s:struct_values_%s" % (corpus, checksum))
                cache_filename_temp = cache_filename + "." + unique_id

                with open(cache_filename_temp, "w") as f:
                    json.dump(result["corpora"][corpus].get(struct, []), f)
                os.rename(cache_filename_temp, cache_filename)

                if "debug" in args:
                    result.setdefault("DEBUG", {})
                    result["DEBUG"].setdefault("caches_saved", [])
                    result["DEBUG"]["caches_saved"].append("%s:%s" % (corpus, struct))

    yield result


################################################################################
# COUNT
################################################################################

@app.route("/count", methods=["GET", "POST"])
@main_handler
def count(args):
    """Perform a CQP query and return a count of the given words/attrs.

    The required parameters are
     - corpus: the CWB corpus
     - cqp: the CQP query string
     - groupby: comma separated list of positional attributes
     - groupby_struct: comma separated list of structural attributes

    The optional parameters are
     - within: only search for matches within the given s-attribute (e.g., within a sentence)
       (default: no within)
     - cut: set cutoff threshold to reduce the size of the result
       (default: no cutoff)
     - ignore_case: changes all values of the selected attribute to lower case
     - incremental: incrementally report the progress while executing
       (default: false)
     - expand_prequeries: when using multiple queries, this determines whether
       subsequent queries should be run on the containing sentences (or any other structural attribute
       defined by 'within') from the previous query, or just the matches.
       (default: true)
    """
    assert_key("cqp", args, r"", True)
    assert_key("corpus", args, IS_IDENT, True)
    assert_key("groupby", args, IS_IDENT, False)
    assert_key("groupby_struct", args, IS_IDENT, False)
    assert_key("cut", args, IS_NUMBER)
    assert_key("ignore_case", args, IS_IDENT)
    assert_key("incremental", args, r"(true|false)")

    incremental = args.get("incremental", "").lower() == "true"

    corpora = parse_corpora(args)

    check_authentication(corpora)

    groupby = args.get("groupby") or []
    if isinstance(groupby, str):
        groupby = groupby.split(QUERY_DELIM)

    groupby_struct = args.get("groupby_struct") or []
    if isinstance(groupby_struct, str):
        groupby_struct = groupby_struct.split(QUERY_DELIM)

    assert groupby or groupby_struct, "Either 'groupby' or 'groupby_struct' needs to be specified."

    groupby = [(g, False) for g in groupby] + [(g, True) for g in groupby_struct]

    ignore_case = args.get("ignore_case") or []
    if isinstance(ignore_case, str):
        ignore_case = ignore_case.split(QUERY_DELIM)
    ignore_case = set(ignore_case)

    within = parse_within(args)

    start = int(args.get("start") or 0)
    end = int(args.get("end") or -1)

    split = args.get("split", "")
    if isinstance(split, str):
        split = split.split(QUERY_DELIM)

    strippointer = args.get("strippointer", "")
    if isinstance(strippointer, str):
        strippointer = strippointer.split(QUERY_DELIM)

    top = args.get("top", "")
    if isinstance(top, str):
        if ":" in top:
            top = dict((x.split(":")[0], int(x.split(":")[1])) for x in top.split(QUERY_DELIM))
        else:
            top = dict((x, 1) for x in top.split(QUERY_DELIM))

    # Sort numbered CQP-queries numerically
    cqp, subcqp = parse_cqp_subcqp(args)
    if subcqp:
        cqp.append(subcqp)

    simple = args.get("simple", "").lower() == "true"

    if cqp == ["[]"]:
        simple = True

    expand_prequeries = not args.get("expand_prequeries", "").lower() == "false"

    result = {"corpora": {}}
    debug = {}
    zero_hits = []
    read_from_cache = 0

    if args["cache"]:
        for corpus in corpora:
            corpus_checksum = get_hash((cqp,
                                        groupby,
                                        within[corpus],
                                        sorted(ignore_case),
                                        expand_prequeries))

            cache_hits_filename = os.path.join(config.CACHE_DIR, "%s:count_size_%s" % (corpus, corpus_checksum))
            if os.path.isfile(cache_hits_filename):
                with open(cache_hits_filename, "r") as f:
                    nr_hits, _ = f.read().split(";", 1)
                read_from_cache += 1
                if nr_hits == "0":
                    zero_hits.append(corpus)

        if "debug" in args:
            debug["cache_coverage"] = "%d/%d" % (read_from_cache, len(corpora))

    total_stats = [{"absolute": defaultdict(int),
                    "relative": defaultdict(float),
                    "sums": {"absolute": 0, "relative": 0.0}} for i in range(len(subcqp) + 1)]

    ns = Namespace()  # To make variables writable from nested functions
    ns.total_size = 0

    count_function = count_query_worker if not simple else count_query_worker_simple

    ns.progress_count = 0
    if incremental:
        yield {"progress_corpora": list(c for c in corpora if c not in zero_hits)}

    for corpus in zero_hits:
        result["corpora"][corpus] = [{"absolute": {},
                                      "relative": {},
                                      "sums": {"absolute": 0, "relative": 0.0}} for i in range(len(subcqp) + 1)]
        for i in range(len(subcqp)):
            result["corpora"][corpus][i + 1]["cqp"] = subcqp[i]

    @reraise_with_stack
    def anti_timeout(queue):
        with ThreadPoolExecutor(max_workers=config.PARALLEL_THREADS) as executor:
            future_query = dict((executor.submit(count_function, corpus=corpus, cqp=cqp, groupby=groupby,
                                                 within=within[corpus], ignore_case=ignore_case,
                                                 expand_prequeries=expand_prequeries,
                                                 use_cache=args["cache"]), corpus)
                                for corpus in corpora if not corpus in zero_hits)

            for future in futures.as_completed(future_query):
                corpus = future_query[future]
                if future.exception() is not None:
                    raise CQPError(future.exception())
                else:
                    lines, nr_hits, corpus_size = future.result()

                    ns.total_size += corpus_size
                    corpus_stats = [{"absolute": defaultdict(int),
                                     "relative": defaultdict(float),
                                     "sums": {"absolute": 0, "relative": 0.0}} for i in range(len(subcqp) + 1)]

                    query_no = 0
                    for line in lines:
                        if line == END_OF_LINE:
                            # EOL means the start of a new subcqp result
                            query_no += 1
                            if subcqp:
                                corpus_stats[query_no]["cqp"] = subcqp[query_no - 1]
                            continue
                        freq, ngram = line.lstrip().split(" ", 1)

                        if len(groupby) > 1:
                            ngram_groups = ngram.split("\t")
                        else:
                            ngram_groups = [ngram]

                        all_ngrams = []

                        for i, ngram in enumerate(ngram_groups):
                            # Split value sets and treat each value as a hit
                            if groupby[i][0] in split:
                                tokens = [t + "|" for t in ngram.split(
                                    "| ")]  # We can't split on just space due to spaces in annotations
                                tokens[-1] = tokens[-1][:-1]
                                if groupby[i][0] in top:
                                    split_tokens = [[x for x in token.split("|") if x][:top[groupby[i][0]]]
                                                    if not token == "|" else ["|"] for token in tokens]
                                else:
                                    split_tokens = [[x for x in token.split("|") if x] if not token == "|" else [""]
                                                    for token in tokens]
                                ngrams = itertools.product(*split_tokens)
                                ngrams = tuple(x for x in ngrams)
                            else:
                                if not groupby[i][1]:
                                    ngrams = (tuple(ngram.split(" ")),)
                                else:
                                    ngrams = (ngram,)

                            # Remove multi word pointers
                            if groupby[i][0] in strippointer:
                                for j in range(len(ngrams)):
                                    if ":" in ngrams[j]:
                                        ngramtemp, pointer = ngrams[j].rsplit(":", 1)
                                        if pointer.isnumeric():
                                            ngrams[j] = ngramtemp

                            all_ngrams.append(ngrams)

                        cross = list(itertools.product(*all_ngrams))

                        for ngram in cross:
                            corpus_stats[query_no]["absolute"][ngram] += int(freq)
                            corpus_stats[query_no]["relative"][ngram] += int(freq) / float(corpus_size) * 1000000
                            corpus_stats[query_no]["sums"]["absolute"] += int(freq)
                            corpus_stats[query_no]["sums"]["relative"] += int(freq) / float(corpus_size) * 1000000
                            total_stats[query_no]["absolute"][ngram] += int(freq)
                            total_stats[query_no]["sums"]["absolute"] += int(freq)

                    result["corpora"][corpus] = corpus_stats

                    if incremental:
                        queue.put({"progress_%d" % ns.progress_count: corpus})
                        ns.progress_count += 1
            queue.put("DONE")

    for msg in prevent_timeout(anti_timeout):
        yield msg

    result["count"] = len(total_stats[0]["absolute"])

    # Calculate relative numbers for the total
    for query_no in range(len(subcqp) + 1):
        if end > -1 and (start > 0 or len(total_stats[0]["absolute"]) > (end - start) + 1):
            # Only a selected range of results requested
            total_absolute = sorted(total_stats[query_no]["absolute"].items(), key=lambda x: x[1],
                                    reverse=True)[start:end + 1]
            for ngram, freq in total_absolute:
                total_stats[query_no]["relative"][ngram] = freq / float(ns.total_size) * 1000000

                for corpus in corpora:
                    new_corpus_part = {"absolute": {}, "relative": {},
                                       "sums": result["corpora"][corpus][query_no]["sums"]}
                    if ngram in result["corpora"][corpus][query_no]["absolute"]:
                        new_corpus_part["absolute"][ngram] = result["corpora"][corpus][query_no]["absolute"][ngram]
                    if ngram in result["corpora"][corpus][query_no]["relative"]:
                        new_corpus_part["relative"][ngram] = result["corpora"][corpus][query_no]["relative"][ngram]

                result["corpora"][corpus][query_no] = new_corpus_part

            total_stats[query_no]["absolute"] = dict(total_absolute)
        else:
            # Complete results requested
            for ngram, freq in total_stats[query_no]["absolute"].items():
                total_stats[query_no]["relative"][ngram] = freq / float(ns.total_size) * 1000000

        for corpus in corpora:
            for relabs in ("absolute", "relative"):
                new_list = []
                for ngram, freq in result["corpora"][corpus][query_no][relabs].items():
                    row = {"value": dict((key[0], ngram[i]) for i, key in enumerate(groupby)),
                           "freq": freq}
                    new_list.append(row)
                result["corpora"][corpus][query_no][relabs] = new_list

        total_stats[query_no]["sums"]["relative"] = (total_stats[query_no]["sums"]["absolute"] / float(ns.total_size)
                                                     * 1000000 if ns.total_size > 0 else 0.0)

        if subcqp and query_no > 0:
            total_stats[query_no]["cqp"] = subcqp[query_no - 1]

        for relabs in ("absolute", "relative"):
            new_list = []
            for ngram, freq in total_stats[query_no][relabs].items():
                row = {"value": dict((key[0], ngram[i]) for i, key in enumerate(groupby)),
                       "freq": freq}
                new_list.append(row)
                total_stats[query_no][relabs] = new_list

    result["total"] = total_stats if len(total_stats) > 1 else total_stats[0]

    if not subcqp:
        for corpus in corpora:
            result["corpora"][corpus] = result["corpora"][corpus][0]

    if "debug" in args:
        debug.update({"cqp": cqp, "simple": simple})
        result["DEBUG"] = debug

    yield result


@app.route("/count_all", methods=["GET", "POST"])
@main_handler
def count_all(args):
    """Return a count of the given attrs.

    The required parameters are
     - corpus: the CWB corpus
     - groupby: positional or structural attributes

    The optional parameters are
     - within: only search for matches within the given s-attribute (e.g., within a sentence)
       (default: no within)
     - cut: set cutoff threshold to reduce the size of the result
       (default: no cutoff)
     - ignore_case: changes all values of the selected attribute to lower case
     - incremental: incrementally report the progress while executing
       (default: false)
    """
    assert_key("corpus", args, IS_IDENT, True)
    assert_key("groupby", args, IS_IDENT, True)
    assert_key("cut", args, IS_NUMBER)
    assert_key("ignore_case", args, IS_IDENT)
    assert_key("incremental", args, r"(true|false)")

    args["cqp"] = "[]"  # Dummy value, not used
    args["simple"] = "true"

    yield generator_to_dict(count(args))


def remap_keys(mapping):
    return [{'key': k, 'value': v} for k, v in mapping.items()]


def strptime(date):
    """Take a date in string format and return a datetime object.
    Input must be on the format "YYYYMMDDhhmmss".
    We need this since the built in strptime isn't thread safe (and this is much faster)."""
    year = int(date[:4])
    month = int(date[4:6]) if len(date) > 4 else 1
    day = int(date[6:8]) if len(date) > 6 else 1
    hour = int(date[8:10]) if len(date) > 8 else 0
    minute = int(date[10:12]) if len(date) > 10 else 0
    second = int(date[12:14]) if len(date) > 12 else 0
    return datetime.datetime(year, month, day, hour, minute, second)


@app.route("/count_time", methods=["GET", "POST"])
@main_handler
def count_time(args):
    """
    """
    assert_key("cqp", args, r"", True)
    assert_key("corpus", args, IS_IDENT, True)
    assert_key("cut", args, IS_NUMBER)
    assert_key("incremental", args, r"(true|false)")
    assert_key("granularity", args, r"[ymdhnsYMDHNS]")
    assert_key("from", args, r"^\d{14}$")
    assert_key("to", args, r"^\d{14}$")
    assert_key("strategy", args, r"^[123]$")

    incremental = args.get("incremental", "").lower() == "true"

    corpora = parse_corpora(args)
    check_authentication(corpora)
    within = parse_within(args)

    # Sort numbered CQP-queries numerically
    cqp, subcqp = parse_cqp_subcqp(args)
    if subcqp:
        cqp.append(subcqp)
    granularity = (args.get("granularity") or "y").lower()
    fromdate = args.get("from", "")
    todate = args.get("to", "")

    # Check that we have a suitable date range for the selected granularity
    df = None
    dt = None

    if fromdate or todate:
        if not fromdate or not todate:
            raise ValueError("When using 'from' or 'to', both need to be specified.")

    # Get date range of selected corpora
    corpus_data = corpus_info({"corpus": QUERY_DELIM.join(corpora), "cache": args["cache"]}, no_combined_cache=True)
    corpora_copy = corpora.copy()

    if fromdate and todate:
        df = strptime(fromdate)
        dt = strptime(todate)

        # Remove corpora not within selected date span
        for c in corpus_data["corpora"]:
            firstdate = corpus_data["corpora"][c]["info"].get("FirstDate")
            lastdate = corpus_data["corpora"][c]["info"].get("LastDate")
            if firstdate and lastdate:
                firstdate = strptime(firstdate.replace("-", "").replace(":", "").replace(" ", ""))
                lastdate = strptime(lastdate.replace("-", "").replace(":", "").replace(" ", ""))

                if not (firstdate <= dt and lastdate >= df):
                    corpora.remove(c)
    else:
        # If no date range was provided, use whole date range of the selected corpora
        for c in corpus_data["corpora"]:
            firstdate = corpus_data["corpora"][c]["info"].get("FirstDate")
            lastdate = corpus_data["corpora"][c]["info"].get("LastDate")
            if firstdate and lastdate:
                firstdate = strptime(firstdate.replace("-", "").replace(":", "").replace(" ", ""))
                lastdate = strptime(lastdate.replace("-", "").replace(":", "").replace(" ", ""))

                if not df or firstdate < df:
                    df = firstdate
                if not dt or lastdate > dt:
                    dt = lastdate

    if df and dt:
        maxpoints = 3600

        if granularity == "y":
            add = relativedelta(years=maxpoints)
        elif granularity == "m":
            add = relativedelta(months=maxpoints)
        elif granularity == "d":
            add = relativedelta(days=maxpoints)
        elif granularity == "h":
            add = relativedelta(hours=maxpoints)
        elif granularity == "n":
            add = relativedelta(minutes=maxpoints)
        elif granularity == "s":
            add = relativedelta(seconds=maxpoints)

        if dt > (df + add):
            raise ValueError("The date range is too large for the selected granularity. "
                             "Use 'to' and 'from' to limit the range.")

    strategy = int(args.get("strategy") or 1)

    if granularity in "hns":
        groupby = [(v, True) for v in ("text_datefrom", "text_timefrom", "text_dateto", "text_timeto")]
    else:
        groupby = [(v, True) for v in ("text_datefrom", "text_dateto")]

    result = {"corpora": {}}
    corpora_sizes = {}

    ns = Namespace()
    total_rows = [[] for i in range(len(subcqp) + 1)]
    ns.total_size = 0

    ns.progress_count = 0
    if incremental:
        yield {"progress_corpora": corpora}

    def anti_timeout(queue):
        with ThreadPoolExecutor(max_workers=config.PARALLEL_THREADS) as executor:
            future_query = dict((executor.submit(count_query_worker, corpus=corpus, cqp=cqp, groupby=groupby,
                                                 within=within, use_cache=args["cache"]), corpus)
                                for corpus in corpora)

            for future in futures.as_completed(future_query):
                corpus = future_query[future]
                if future.exception() is not None:
                    if "Can't find attribute ``text_datefrom''" not in str(future.exception()):
                        raise CQPError(future.exception())
                else:
                    lines, _, corpus_size = future.result()

                    corpora_sizes[corpus] = corpus_size
                    ns.total_size += corpus_size

                    query_no = 0
                    for line in lines:
                        if line == END_OF_LINE:
                            query_no += 1
                            continue
                        count, values = line.lstrip().split(" ", 1)
                        values = values.strip(" ")
                        if granularity in "hns":
                            datefrom, timefrom, dateto, timeto = values.split("\t")
                            # Only use the value from the first token
                            timefrom = timefrom.split(" ")[0]
                            timeto = timeto.split(" ")[0]
                        else:
                            datefrom, dateto = values.split("\t")
                            timefrom = ""
                            timeto = ""

                        # Only use the value from the first token
                        datefrom = datefrom.split(" ")[0]
                        dateto = dateto.split(" ")[0]

                        total_rows[query_no].append({"corpus": corpus, "df": datefrom + timefrom, "dt": dateto + timeto,
                                                     "sum": int(count)})

                if incremental:
                    queue.put({"progress_%d" % ns.progress_count: corpus})
                    ns.progress_count += 1
            queue.put("DONE")

    for msg in prevent_timeout(anti_timeout):
        yield msg

    corpus_timedata = generator_to_dict(timespan({"corpus": corpora, "granularity": granularity, "from": fromdate,
                                                  "to": todate, "strategy": str(strategy), "cache": args["cache"]},
                                                 no_combined_cache=True))
    search_timedata = []
    search_timedata_combined = []
    for total_row in total_rows:
        temp = timespan_calculator(total_row, granularity=granularity, strategy=strategy)
        search_timedata.append(temp["corpora"])
        search_timedata_combined.append(temp["combined"])

    for corpus in corpora:
        corpus_stats = [{"absolute": defaultdict(int),
                         "relative": defaultdict(float),
                         "sums": {"absolute": 0, "relative": 0.0}} for i in range(len(subcqp) + 1)]

        basedates = dict([(date, None if corpus_timedata["corpora"][corpus][date] == 0 else 0)
                          for date in corpus_timedata["corpora"].get(corpus, {})])

        for i, s in enumerate(search_timedata):

            prevdate = None
            for basedate in sorted(basedates):
                if not basedates[basedate] == prevdate:
                    corpus_stats[i]["absolute"][basedate] = basedates[basedate]
                    corpus_stats[i]["relative"][basedate] = basedates[basedate]
                prevdate = basedates[basedate]

            for row in s.get(corpus, {}).items():
                date, count = row
                corpus_date_size = float(corpus_timedata["corpora"].get(corpus, {}).get(date, 0))
                if corpus_date_size > 0.0:
                    corpus_stats[i]["absolute"][date] += count
                    corpus_stats[i]["relative"][date] += (count / corpus_date_size * 1000000)
                    corpus_stats[i]["sums"]["absolute"] += count
                    corpus_stats[i]["sums"]["relative"] += (count / corpus_date_size * 1000000)

            if subcqp and i > 0:
                corpus_stats[i]["cqp"] = subcqp[i - 1]

        result["corpora"][corpus] = corpus_stats if len(corpus_stats) > 1 else corpus_stats[0]

    total_stats = [{"absolute": defaultdict(int),
                    "relative": defaultdict(float),
                    "sums": {"absolute": 0, "relative": 0.0}} for i in range(len(subcqp) + 1)]

    basedates = dict([(date, None if corpus_timedata["combined"][date] == 0 else 0)
                      for date in corpus_timedata.get("combined", {})])

    for i, s in enumerate(search_timedata_combined):

        prevdate = None
        for basedate in sorted(basedates):
            if not basedates[basedate] == prevdate:
                total_stats[i]["absolute"][basedate] = basedates[basedate]
                total_stats[i]["relative"][basedate] = basedates[basedate]
            prevdate = basedates[basedate]

        if s:
            for row in s.items():
                date, count = row
                combined_date_size = float(corpus_timedata["combined"].get(date, 0))
                if combined_date_size > 0.0:
                    total_stats[i]["absolute"][date] += count
                    total_stats[i]["relative"][date] += (
                        count / combined_date_size * 1000000) if combined_date_size else 0
                    total_stats[i]["sums"]["absolute"] += count

        total_stats[i]["sums"]["relative"] = total_stats[i]["sums"]["absolute"] / float(
            ns.total_size) * 1000000 if ns.total_size > 0 else 0.0
        if subcqp and i > 0:
            total_stats[i]["cqp"] = subcqp[i - 1]

    result["combined"] = total_stats if len(total_stats) > 1 else total_stats[0]

    # Add zero values for the corpora we removed because of the selected date span
    for corpus in set(corpora_copy).difference(set(corpora)):
        result["corpora"][corpus] = {"absolute": 0, "relative": 0.0, "sums": {"absolute": 0, "relative": 0.0}}

    if "debug" in args:
        result["DEBUG"] = {"cqp": cqp}

    yield result


def count_query_worker(corpus, cqp, groupby, within, ignore_case=[], cut=None, expand_prequeries=True, use_cache=False):
    subcqp = None
    if isinstance(cqp[-1], list):
        subcqp = cqp[-1]
        cqp = cqp[:-1]

    if use_cache:
        checksum = get_hash((cqp,
                             groupby,
                             within,
                             sorted(ignore_case),
                             expand_prequeries))

        unique_id = str(uuid.uuid4())

        cache_filename = os.path.join(config.CACHE_DIR, "%s:count_%s" % (corpus, checksum))
        cache_filename_temp = cache_filename + "." + unique_id

        cache_size_filename = os.path.join(config.CACHE_DIR, "%s:count_size_%s" % (corpus, checksum))
        cache_size_filename_temp = cache_size_filename + "." + unique_id

        is_cached = os.path.isfile(cache_filename) and os.path.isfile(cache_size_filename)
        if is_cached:
            with open(cache_size_filename, "r") as f:
                corpus_hits, corpus_size = (int(v) for v in f.read().split(";", 1))
            if corpus_hits == 0:
                return [END_OF_LINE] * len(subcqp) if subcqp else [], corpus_hits, corpus_size
            with open(cache_filename, "rb") as f:
                return pickle.load(f), corpus_hits, corpus_size
    else:
        is_cached = False

    do_optimize = True
    cqpparams = {"within": within,
                 "cut": cut}

    cmd = ["%s;" % corpus]
    for i, c in enumerate(cqp):
        cqpparams_temp = cqpparams.copy()
        pre_query = i + 1 < len(cqp)

        if pre_query and expand_prequeries:
            cqpparams_temp["expand"] = "to " + cqpparams["within"]

        if do_optimize:
            cmd += query_optimize(c, cqpparams_temp, find_match=(not pre_query))[1]
        else:
            cmd += make_query(make_cqp(c, **cqpparams_temp))

        if pre_query:
            cmd += ["Last;"]

    cmd += ["size Last;"]
    cmd += ["info; .EOL.;"]

    # TODO: Match targets in a better way
    has_target = any("@[" in x for x in cqp)

    cmd += ["""tabulate Last %s > "| sort | uniq -c | sort -nr";""" % ", ".join("%s %s%s" % (
        "target" if has_target else ("match" if g[1] else "match .. matchend"), g[0], " %c" if g in ignore_case else "") for g in groupby)]

    if subcqp:
        cmd += ["mainresult=Last;"]
        if "expand" in cqpparams_temp:
            del cqpparams_temp["expand"]
        for c in subcqp:
            cmd += [".EOL.;"]
            cmd += ["mainresult;"]
            cmd += query_optimize(c, cqpparams_temp, find_match=True)[1]
            cmd += ["""tabulate Last %s > "| sort | uniq -c | sort -nr";""" % ", ".join(
                "match .. matchend %s" % g[0] for g in groupby)]

    cmd += ["exit;"]

    lines = run_cqp(cmd)

    # Skip CQP version
    next(lines)

    # Size of the query result
    nr_hits = int(next(lines))

    # Get corpus size
    for line in lines:
        if line.startswith("Size:"):
            _, corpus_size = line.split(":")
            corpus_size = int(corpus_size.strip())
        elif line == END_OF_LINE:
            break

    if use_cache and not is_cached:
        with open(cache_size_filename_temp, "w") as f:
            f.write("%d;%d" % (nr_hits, corpus_size))
        os.rename(cache_size_filename_temp, cache_size_filename)

        # Only save actual data if number of lines doesn't exceed the limit
        if nr_hits <= config.CACHE_MAX_STATS:
            lines = tuple(lines)
            with open(cache_filename_temp, "wb") as f:
                pickle.dump(lines, f, protocol=-1)
            os.rename(cache_filename_temp, cache_filename)

    return lines, nr_hits, corpus_size


def count_query_worker_simple(corpus, cqp, groupby, within=None, ignore_case=[], expand_prequeries=True, use_cache=False):
    """Worker for simple statistics queries which can be run using cwb-scan-corpus.
    Currently only used for searches on [] (any word)."""

    lines = list(run_cwb_scan(corpus, [g[0] for g in groupby]))
    nr_hits = 0

    ic_index = []
    new_lines = {}
    if ignore_case:
        ic_index = [i for i, g in enumerate(groupby) if g[0] in ignore_case]

    for i in range(len(lines)):
        c, v = lines[i].split("\t", 1)
        nr_hits += int(c)

        if ic_index:
            v = "\t".join(vv.lower() if i in ic_index else vv for i, vv in enumerate(v.split("\t")))
            new_lines[v] = new_lines.get(v, 0) + int(c)
        else:
            # Convert result to the same format as the regular CQP count
            lines[i] = "%s %s" % (c, v)

    if ic_index:
        lines = []
        for v, c in new_lines.items():
            # Convert result to the same format as the regular CQP count
            lines.append("%s %s" % (c, v))

    # Corpus size equals number of hits since we count all tokens
    corpus_size = nr_hits
    return lines, nr_hits, corpus_size


@app.route("/loglike", methods=["GET", "POST"])
@main_handler
def loglike(args):
    """Run a log-likelihood comparison on two queries.

    The required parameters are
     - set1_cqp: the first CQP query
     - set2_cqp: the second CQP query
     - set1_corpus: the corpora for the first query
     - set2_corpus: the corpora for the second query
     - groupby: what positional or structural attribute to use for comparison

    The optional parameters are
     - ignore_case: ignore case when comparing
     - max: maxium number of results per set
       (default: 15)
    """

    def expected(total, wordtotal, sumtotal):
        """ The expected is that the words are uniformely distributed over the corpora. """
        return wordtotal * (float(total) / sumtotal)

    def compute_loglike(wf1_tot1, wf2_tot2):
        """ Compute log-likelihood for a single pair. """
        wf1, tot1 = wf1_tot1
        wf2, tot2 = wf2_tot2
        e1 = expected(tot1, wf1 + wf2, tot1 + tot2)
        e2 = expected(tot2, wf1 + wf2, tot1 + tot2)
        (l1, l2) = (0, 0)
        if wf1 > 0:
            l1 = wf1 * math.log(wf1 / e1)
        if wf2 > 0:
            l2 = wf2 * math.log(wf2 / e2)
        loglike = 2 * (l1 + l2)
        return round(loglike, 2)

    def critical(val):
        # 95th percentile; 5% level; p < 0.05; critical value = 3.84
        # 99th percentile; 1% level; p < 0.01; critical value = 6.63
        # 99.9th percentile; 0.1% level; p < 0.001; critical value = 10.83
        # 99.99th percentile; 0.01% level; p < 0.0001; critical value = 15.13
        return val > 15.13

    def select(w, ls):
        """ Split annotations on | and returns as list. If annotation is missing, returns the word instead. """
        #    for c in w:
        #        if not (c.isalpha() or (len(w) > 1 and c in '-:')):
        #            return []
        xs = [l for l in ls.split('|') if len(l) > 0]
        return xs or [w]

    def wf_frequencies(texts):
        freqs = []
        for (name, text) in texts:
            d = defaultdict(int)  # Lemgram frequency
            tc = 0  # Total number of tokens
            for w in [r for s in text for (w, a) in s for r in select(w, a['lex'])]:
                tc += 1
                d[w] += 1
            freqs.append((name, d, tc))
        return freqs

    def reference_material(filename):
        d = defaultdict(int)
        tot = 0
        with open(filename, encoding='utf8') as f:
            for l in f:
                (wf, msd, lemgram, comp, af, rf) = l[:-1].split('\t')
                for ll in select(wf, lemgram):
                    tot += int(af)  # Total number of tokens
                    d[ll] += int(af)  # Lemgram frequency
        return d, tot

    def compute_list(d1, tot1, ref, reftot):
        """ Compute log-likelyhood for lists. """
        result = []
        all_w = set(d1.keys()).union(set(ref.keys()))
        for w in all_w:
            ll = compute_loglike((d1.get(w, 0), tot1), (ref.get(w, 0), reftot))
            result.append((ll, w))
        result.sort(reverse=True)
        return result

    def compute_ll_stats(ll_list, count, sets):
        """ Calculate max, min, average, and truncates word list. """
        tot = len(ll_list)
        new_list = []

        set1count, set2count = 0, 0
        for ll_w in ll_list:
            ll, w = ll_w

            if (sets[0]["freq"].get(w) and not sets[1]["freq"].get(w)) or sets[0]["freq"].get(w) and (
                sets[0]["freq"].get(w, 0) / (sets[0]["total"] * 1.0)) > (
                sets[1]["freq"].get(w, 0) / (sets[1]["total"] * 1.0)):
                set1count += 1
                if set1count <= count or not count:
                    new_list.append((ll * -1, w))
            else:
                set2count += 1
                if set2count <= count or not count:
                    new_list.append((ll, w))

            if count and (set1count >= count and set2count >= count):
                break

        nums = [ll for (ll, _) in ll_list]
        return (
            new_list,
            round(sum(nums) / float(tot), 2) if tot else 0.0,
            min(nums) if nums else 0.0,
            max(nums) if nums else 0.0
        )

    assert_key("set1_cqp", args, r"", True)
    assert_key("set2_cqp", args, r"", True)
    assert_key("set1_corpus", args, r"", True)
    assert_key("set2_corpus", args, r"", True)
    assert_key("groupby", args, IS_IDENT, True)
    assert_key("ignore_case", args, IS_IDENT)
    assert_key("max", args, IS_NUMBER, False)

    maxresults = int(args.get("max") or 15)

    set1 = args.get("set1_corpus").upper()
    if isinstance(set1, str):
        set1 = set1.split(QUERY_DELIM)
    set1 = set(set1)
    set2 = args.get("set2_corpus").upper()
    if isinstance(set2, str):
        set2 = set2.split(QUERY_DELIM)
    set2 = set(set2)

    corpora = set1.union(set2)
    check_authentication(corpora)

    same_cqp = args.get("set1_cqp") == args.get("set2_cqp")

    result = {}

    def anti_timeout(queue):
        # If same CQP for both sets, handle as one query for better performance
        if same_cqp:
            args["cqp"] = args.get("set1_cqp")
            args["corpus"] = QUERY_DELIM.join(corpora)
            count_result = generator_to_dict(count(args))

            sets = [{"total": 0, "freq": defaultdict(int)}, {"total": 0, "freq": defaultdict(int)}]
            for i, cset in enumerate((set1, set2)):
                for corpus in cset:
                    sets[i]["total"] += count_result["corpora"][corpus]["sums"]["absolute"]
                    if len(cset) == 1:
                        sets[i]["freq"] = dict((tuple((y[0], tuple(y[1])) for y in sorted(x["value"].items())), x["freq"])
                                               for x in count_result["corpora"][corpus]["absolute"])
                    else:
                        for w, f in ((tuple((y[0], tuple(y[1])) for y in sorted(x["value"].items())), x["freq"])
                                     for x in count_result["corpora"][corpus]["absolute"]):
                            sets[i]["freq"][w] += f

        else:
            args1, args2 = args.copy(), args.copy()
            args1["corpus"] = QUERY_DELIM.join(set1)
            args1["cqp"] = args.get("set1_cqp")
            args2["corpus"] = QUERY_DELIM.join(set2)
            args2["cqp"] = args.get("set2_cqp")
            count_result = [generator_to_dict(count(args1)), generator_to_dict(count(args2))]

            sets = [{}, {}]
            for i, cset in enumerate((set1, set2)):
                count_result_temp = count_result if same_cqp else count_result[i]
                sets[i]["total"] = count_result_temp["total"]["sums"]["absolute"]
                sets[i]["freq"] = dict((tuple((y[0], tuple(y[1])) for y in sorted(x["value"].items())), x["freq"])
                                       for x in count_result_temp["total"]["absolute"])

        ll_list = compute_list(sets[0]["freq"], sets[0]["total"], sets[1]["freq"], sets[1]["total"])
        (ws, avg, mi, ma) = compute_ll_stats(ll_list, maxresults, sets)

        result["loglike"] = {}
        result["average"] = avg
        result["set1"] = {}
        result["set2"] = {}

        for (ll, w) in ws:
            w_formatted = " ".join(w[0][1])
            result["loglike"][w_formatted] = ll
            result["set1"][w_formatted] = sets[0]["freq"].get(w, 0)
            result["set2"][w_formatted] = sets[1]["freq"].get(w, 0)

        queue.put("DONE")

    for msg in prevent_timeout(anti_timeout):
        yield msg

    yield result


################################################################################
# LEMGRAM_COUNT
################################################################################

@app.route("/lemgram_count", methods=["GET", "POST"])
@main_handler
def lemgram_count(args):
    """Return lemgram statistics per corpus.

    The required parameters are
     - lemgram: list of lemgrams

    The optional parameters are
     - corpus: the CWB corpus/corpora
       (default: all corpora)
     - count: what to count (lemgram/prefix/suffix)
       (default: lemgram)
    """
    assert_key("lemgram", args, r"", True)
    assert_key("corpus", args, IS_IDENT)
    assert_key("count", args, r"(lemgram|prefix|suffix)")

    corpora = parse_corpora(args)

    check_authentication(corpora)

    lemgram = args.get("lemgram")
    if isinstance(lemgram, str):
        lemgram = lemgram.split(QUERY_DELIM)
    lemgram = set(lemgram)

    count = args.get("count") or "lemgram"
    if isinstance(count, str):
        count = count.split(QUERY_DELIM)
    count = set(count)

    counts = {"lemgram": "freq",
              "prefix": "freq_prefix",
              "suffix": "freq_suffix"}

    sums = " + ".join("SUM(%s)" % counts[c] for c in count)

    lemgram_sql = " lemgram IN (%s)" % ", ".join("'%s'" % sql_escape(l) for l in lemgram)
    corpora_sql = " AND corpus IN (%s)" % ", ".join("'%s'" % sql_escape(c) for c in corpora) if corpora else ""

    sql = "SELECT lemgram, " + sums + " AS freq FROM lemgram_index WHERE" + lemgram_sql + corpora_sql + \
          " GROUP BY lemgram COLLATE utf8_bin;"

    result = {}
    cursor = mysql.connection.cursor()
    cursor.execute(sql)

    for row in cursor:
        # We need this check here, since a search for "hår" also returns "här" and "har".
        if row["lemgram"] in lemgram and int(row["freq"]) > 0:
            result[row["lemgram"]] = int(row["freq"])

    yield result


def sql_escape(s):
    with app.app_context():
        return mysql.connection.escape_string(s).decode("utf-8") if isinstance(s, str) else s


################################################################################
# TIMESPAN
################################################################################

@app.route("/timespan", methods=["GET", "POST"])
@main_handler
def timespan(args, no_combined_cache=False):
    """Calculate timespan information for corpora.
    The time information is retrieved from the database.

    The required parameters are
     - corpus: the CWB corpus/corpora

    The optional parameters are
     - granularity: granularity of result (y = year, m = month, d = day, h = hour, n = minute, s = second)
       (default: year)
     - combined: include combined results
       (default: true)
     - per_corpus: include results per corpus
       (default: true)
     - from: from this date and time, on the format 20150513063500 or 2015-05-13 06:35:00 (times optional) (optional)
     - to: to this date and time (optional)
    """
    assert_key("corpus", args, IS_IDENT, True)
    assert_key("granularity", args, r"[ymdhnsYMDHNS]")
    assert_key("combined", args, r"(true|false)")
    assert_key("per_corpus", args, r"(true|false)")
    assert_key("strategy", args, r"^[123]$")
    assert_key("from", args, r"^(\d{8}\d{6}?|\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?)$")
    assert_key("to", args, r"^(\d{8}\d{6}?|\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?)$")

    corpora = parse_corpora(args)
    # check_authentication(corpora)

    granularity = (args.get("granularity") or "y").lower()
    combined = (not args.get("combined", "").lower() == "false")
    per_corpus = (not args.get("per_corpus", "").lower() == "false")
    strategy = int(args.get("strategy") or 1)
    fromdate = args.get("from")
    todate = args.get("to")

    if fromdate or todate:
        if not fromdate or not todate:
            raise ValueError("When using 'from' or 'to', both need to be specified.")

    shorten = {"y": 4, "m": 7, "d": 10, "h": 13, "n": 16, "s": 19}

    cached_data = []
    corpora_rest = corpora[:]

    if args["cache"]:
        # Check if whole query is cached
        combined_checksum = get_hash((granularity,
                                      combined,
                                      per_corpus,
                                      fromdate,
                                      todate,
                                      sorted(corpora)))
        cache_filename_combined = os.path.join(config.CACHE_DIR, "timespan_%s" % (get_hash(combined_checksum)))
        if os.path.isfile(cache_filename_combined):
            with open(cache_filename_combined, "rb") as f:
                result = pickle.load(f)
            if "debug" in args:
                result.setdefault("DEBUG", {})
                result["DEBUG"]["cache_read"] = True
            yield result
            return

        # Look for per-corpus caches
        for corpus in corpora:
            corpus_checksum = get_hash((fromdate, todate, granularity, strategy))
            cache_filename = os.path.join(config.CACHE_DIR, "%s:timespan_%s" % (corpus, corpus_checksum))

            if os.path.isfile(cache_filename):
                with open(cache_filename, "rb") as f:
                    cached_data.extend(pickle.load(f))
                corpora_rest.remove(corpus)

    ns = {}

    def anti_timeout_fun(queue):
        with app.app_context():
            if corpora_rest:
                corpora_sql = "(%s)" % ", ".join("'%s'" % sql_escape(c) for c in corpora_rest)
                fromto = ""

                if strategy == 1:
                    if fromdate and todate:
                        fromto = " AND ((datefrom >= %s AND dateto <= %s) OR (datefrom <= %s AND dateto >= %s))" % (
                            sql_escape(fromdate), sql_escape(todate), sql_escape(fromdate), sql_escape(todate))
                elif strategy == 2:
                    if todate:
                        fromto += " AND datefrom <= '%s'" % sql_escape(todate)
                    if fromdate:
                        fromto = " AND dateto >= '%s'" % sql_escape(fromdate)
                elif strategy == 3:
                    if fromdate:
                        fromto = " AND datefrom >= '%s'" % sql_escape(fromdate)
                    if todate:
                        fromto += " AND dateto <= '%s'" % sql_escape(todate)

                # TODO: Skip grouping on corpus when we only are after the combined results.
                # We do the granularity truncation and summation in the DB query if we can (depending on strategy),
                # since it's much faster than doing it afterwards

                timedata_corpus = "timedata_date" if granularity in ("y", "m", "d") else "timedata"
                if strategy == 1:
                    # We need the full dates for this strategy, so no truncating of the results
                    sql = "SELECT corpus, datefrom AS df, dateto AS dt, SUM(tokens) AS sum FROM " + timedata_corpus + \
                          " WHERE corpus IN " + corpora_sql + fromto + " GROUP BY corpus, df, dt ORDER BY NULL;"
                else:
                    sql = "SELECT corpus, LEFT(datefrom, " + str(shorten[granularity]) + ") AS df, LEFT(dateto, " + \
                          str(shorten[granularity]) + ") AS dt, SUM(tokens) AS sum FROM " + timedata_corpus + \
                          " WHERE corpus IN " + corpora_sql + fromto + " GROUP BY corpus, df, dt ORDER BY NULL;"
                cursor = mysql.connection.cursor()
                cursor.execute(sql)
            else:
                cursor = tuple()

            if args["cache"]:
                unique_id = str(uuid.uuid4())

                def save_cache(corpus, data):
                    corpus_checksum = get_hash((fromdate, todate, granularity, strategy))
                    cache_filename = os.path.join(config.CACHE_DIR, "%s:timespan_%s" % (corpus, corpus_checksum))
                    cache_filename_temp = cache_filename + "." + unique_id
                    if not os.path.isfile(cache_filename_temp):
                        with open(cache_filename_temp, "wb") as f:
                            pickle.dump(data, f, protocol=-1)
                        os.rename(cache_filename_temp, cache_filename)

                corpus = None
                corpus_data = []
                for row in cursor:
                    if corpus is None:
                        corpus = row["corpus"]
                    elif not row["corpus"] == corpus:
                        save_cache(corpus, corpus_data)
                        corpus_data = []
                        corpus = row["corpus"]
                    corpus_data.append(row)
                    cached_data.append(row)
                if corpus is not None:
                    save_cache(corpus, corpus_data)

            ns["result"] = timespan_calculator(itertools.chain(cached_data, cursor), granularity=granularity,
                                               combined=combined, per_corpus=per_corpus, strategy=strategy)

            if args["cache"] and not no_combined_cache:
                # Save cache for whole query
                cache_filename_combined_temp = "%s.%s" % (cache_filename_combined, unique_id)
                with open(cache_filename_combined_temp, "wb") as f:
                    pickle.dump(ns["result"], f, protocol=-1)
                os.rename(cache_filename_combined_temp, cache_filename_combined)

            queue.put("DONE")

    for msg in prevent_timeout(anti_timeout_fun):
        yield msg

    yield ns["result"]


def timespan_calculator(timedata, granularity="y", combined=True, per_corpus=True, strategy=1):
    """Calculate timespan information for corpora.

    The required parameters are
     - timedata: the time data to be processed

    The optional parameters are
     - granularity: granularity of result (y = year, m = month, d = day, h = hour, n = minute, s = second)
       (default: year)
     - combined: include combined results
       (default: true)
     - per_corpus: include results per corpus
       (default: true)
    """

    gs = {"y": 4, "m": 6, "d": 8, "h": 10, "n": 12, "s": 14}

    def strftime(dt, fmt):
        """Python datetime.strftime < 1900 workaround, taken from https://gist.github.com/2000837"""

        TEMPYEAR = 9996  # We need to use a leap year to support feb 29th

        if dt.year < 1900:
            # Create a copy of this datetime, just in case, then set the year to
            # something acceptable, then replace that year in the resulting string
            tmp_dt = datetime.datetime(TEMPYEAR, dt.month, dt.day,
                                       dt.hour, dt.minute,
                                       dt.second, dt.microsecond,
                                       dt.tzinfo)

            tmp_fmt = fmt
            tmp_fmt = re.sub('(?<!%)((?:%%)*)(%y)', '\\1\x11\x11', tmp_fmt, re.U)
            tmp_fmt = re.sub('(?<!%)((?:%%)*)(%Y)', '\\1\x12\x12\x12\x12', tmp_fmt, re.U)
            tmp_fmt = tmp_fmt.replace(str(TEMPYEAR), '\x13\x13\x13\x13')
            tmp_fmt = tmp_fmt.replace(str(TEMPYEAR)[-2:], '\x14\x14')

            result = tmp_dt.strftime(tmp_fmt)

            if '%c' in fmt:
                # Local datetime format - uses full year but hard for us to guess where.
                result = result.replace(str(TEMPYEAR), str(dt.year))

            result = result.replace('\x11\x11', str(dt.year)[-2:])
            result = result.replace('\x12\x12\x12\x12', str(dt.year))
            result = result.replace('\x13\x13\x13\x13', str(TEMPYEAR))
            result = result.replace('\x14\x14', str(TEMPYEAR)[-2:])

            return result

        else:
            return dt.strftime(fmt)

    def plusminusone(date, value, df, negative=False):
        date = "0" + date if len(date) % 2 else date  # Handle years with three digits
        d = strptime(date)
        if negative:
            d = d - value
        else:
            d = d + value
        return int(strftime(d, df))

    def shorten(date, g):
        alt = 1 if len(date) % 2 else 0  # Handle years with three digits
        return int(date[:gs[g] - alt])

    if granularity == "y":
        df = "%Y"
        add = relativedelta(years=1)
    elif granularity == "m":
        df = "%Y%m"
        add = relativedelta(months=1)
    elif granularity == "d":
        df = "%Y%m%d"
        add = relativedelta(days=1)
    elif granularity == "h":
        df = "%Y%m%d%H"
        add = relativedelta(hours=1)
    elif granularity == "n":
        df = "%Y%m%d%H%M"
        add = relativedelta(minutes=1)
    elif granularity == "s":
        df = "%Y%m%d%H%M%S"
        add = relativedelta(seconds=1)

    rows = defaultdict(list)
    nodes = defaultdict(set)

    datemin = "00000101" if granularity in ("y", "m", "d") else "00000101000000"
    datemax = "99991231" if granularity in ("y", "m", "d") else "99991231235959"

    for row in timedata:
        corpus = row["corpus"]
        datefrom = "".join(x for x in str(row["df"]) if x.isdigit()) if row["df"] else ""
        if datefrom == "0" * len(datefrom):
            datefrom = ""
        dateto = "".join(x for x in str(row["dt"]) if x.isdigit()) if row["dt"] else ""
        if dateto == "0" * len(dateto):
            dateto = ""
        datefrom_short = shorten(datefrom, granularity) if datefrom else 0
        dateto_short = shorten(dateto, granularity) if dateto else 0

        if strategy == 1:
            # Some overlaps permitted
            # (t1 >= t1' AND t2 <= t2') OR (t1 <= t1' AND t2 >= t2')
            if not datefrom_short == dateto_short:
                if not datefrom[gs[granularity]:] == datemin[gs[granularity]:]:
                    # Add 1 to datefrom_short
                    datefrom_short = plusminusone(str(datefrom_short), add, df)

                if not dateto[gs[granularity]:] == datemax[gs[granularity]:]:
                    # Subtract 1 from dateto_short
                    dateto_short = plusminusone(str(dateto_short), add, df, negative=True)

                # Check that datefrom is still before dateto
                if not datefrom < dateto:
                    continue
        elif strategy == 2:
            # All overlaps permitted
            # t1 <= t2' AND t2 >= t1'
            pass
        elif strategy == 3:
            # Strict matching. No overlaps tolerated.
            # t1 >= t1' AND t2 <= t2'

            if not datefrom_short == dateto_short:
                continue

        r = {"datefrom": datefrom_short, "dateto": dateto_short, "corpus": corpus, "freq": int(row["sum"])}
        if combined:
            rows["__combined__"].append(r)
            nodes["__combined__"].add(("f", datefrom_short))
            nodes["__combined__"].add(("t", dateto_short))
        if per_corpus:
            rows[corpus].append(r)
            nodes[corpus].add(("f", datefrom_short))
            nodes[corpus].add(("t", dateto_short))

    corpusnodes = dict((k, sorted(v, key=lambda x: (x[1] if x[1] else 0, x[0])))
                       for k, v in nodes.items())
    result = {}
    if per_corpus:
        result["corpora"] = {}
    if combined:
        result["combined"] = {}

    for corpus, nodes in corpusnodes.items():
        data = defaultdict(int)

        for i in range(0, len(nodes) - 1):
            start = nodes[i]
            end = nodes[i + 1]
            if start[0] == "t":
                start = plusminusone(str(start[1]), add, df) if start[1] else 0
                if start == end[1] and end[0] == "f":
                    continue
            else:
                start = start[1]

            if not end[1]:
                end = 0
            else:
                end = end[1] if end[0] == "t" else plusminusone(str(end[1]), add, df, True)

            if start:
                data["%d" % start] = 0

            for row in rows[corpus]:
                if row["datefrom"] <= start and row["dateto"] >= end:
                    data[str(start if start else "")] += row["freq"]

            if end:
                data["%d" % plusminusone(str(end), add, df, False)] = 0

        if combined and corpus == "__combined__":
            result["combined"] = data
        else:
            result["corpora"][corpus] = data

    return result


################################################################################
# RELATIONS
################################################################################

@app.route("/relations", methods=["GET", "POST"])
@main_handler
def relations(args):
    """Calculate word picture data.

    The required parameters are
     - corpus: the CWB corpus/corpora
     - word: a word or lemgram to lookup

    The optional parameters are
     - min: cut off results with a frequency lower than this
       (default: no cut-off)
     - max: maximum number of results
       (default: 15)
     - type: type of search (word/lemgram)
       (default: word)
     - incremental: incrementally report the progress while executing
       (default: false)
    """
    assert_key("corpus", args, IS_IDENT, True)
    assert_key("word", args, "", True)
    assert_key("type", args, r"(word|lemgram)", False)
    assert_key("min", args, IS_NUMBER, False)
    assert_key("max", args, IS_NUMBER, False)
    assert_key("incremental", args, r"(true|false)")

    corpora = parse_corpora(args)

    check_authentication(corpora)

    incremental = args.get("incremental", "").lower() == "true"

    word = args.get("word")
    search_type = args.get("type", "")
    minfreq = args.get("min")
    sortby = args.get("sortby") or "mi"
    maxresults = int(args.get("max") or 15)
    minfreqsql = " AND freq >= %s" % minfreq if minfreq else ""

    result = {}

    cursor = mysql.connection.cursor()
    cursor.execute("SET @@session.long_query_time = 1000;")

    # Get available tables
    cursor.execute("SHOW TABLES LIKE '" + config.DBWPTABLE + "_%';")
    tables = set(list(x.values())[0] for x in cursor)
    # Filter out corpora which don't exist in database
    corpora = [x for x in corpora if config.DBWPTABLE + "_" + x.upper() in tables]
    if not corpora:
        yield {}
        return

    relations_data = []
    corpora_rest = corpora[:]

    if args["cache"]:
        for corpus in corpora:
            corpus_checksum = get_hash((word,
                                        search_type,
                                        minfreq))
            cache_filename = os.path.join(config.CACHE_DIR, "%s:relations_%s" % (corpus, corpus_checksum))
            if os.path.isfile(cache_filename):
                with open(cache_filename, "rb") as f:
                    relations_data.extend(pickle.load(f))
                corpora_rest.remove(corpus)

    selects = []

    if search_type == "lemgram":
        lemgram_sql = "'%s'" % sql_escape(word)

        for corpus in corpora_rest:
            corpus_sql = "'%s'" % sql_escape(corpus).upper()
            corpus_table = config.DBWPTABLE + "_" + corpus.upper()

            selects.append((corpus.upper(),
                            "(SELECT S1.string AS head, S1.pos AS headpos, F.rel, S2.string AS dep, S2.pos AS deppos, S2.stringextra AS depextra, F.freq, R.freq AS rel_freq, HR.freq AS head_rel_freq, DR.freq AS dep_rel_freq, " + corpus_sql + " AS corpus, F.id " +
                            "FROM `" + corpus_table + "_strings` AS S1, `" + corpus_table + "_strings` AS S2, `" + corpus_table + "` AS F, `" + corpus_table + "_rel` AS R, `" + corpus_table + "_head_rel` AS HR, `" + corpus_table + "_dep_rel` AS DR " +
                            "WHERE S1.string = " + lemgram_sql + " COLLATE utf8_bin AND F.head = S1.id AND S2.id = F.dep " +
                            minfreqsql +
                            "AND F.bfhead = 1 AND F.bfdep = 1 AND F.rel = R.rel AND F.head = HR.head AND F.rel = HR.rel AND F.dep = DR.dep AND F.rel = DR.rel)"
                            ))
            selects.append((None,
                            "(SELECT S1.string AS head, S1.pos AS headpos, F.rel, S2.string AS dep, S2.pos AS deppos, S2.stringextra AS depextra, F.freq, R.freq AS rel_freq, HR.freq AS head_rel_freq, DR.freq AS dep_rel_freq, " + corpus_sql + " AS corpus, F.id " +
                            "FROM `" + corpus_table + "_strings` AS S1, `" + corpus_table + "_strings` AS S2, `" + corpus_table + "` AS F, `" + corpus_table + "_rel` AS R, `" + corpus_table + "_head_rel` AS HR, `" + corpus_table + "_dep_rel` AS DR " +
                            "WHERE S2.string = " + lemgram_sql + " COLLATE utf8_bin AND F.dep = S2.id AND S1.id = F.head " +
                            minfreqsql +
                            "AND F.bfhead = 1 AND F.bfdep = 1 AND F.rel = R.rel AND F.head = HR.head AND F.rel = HR.rel AND F.dep = DR.dep AND F.rel = DR.rel)"
                            ))
    else:
        word_sql = "'%s'" % sql_escape(word)
        word = word

        for corpus in corpora_rest:
            corpus_sql = "'%s'" % sql_escape(corpus).upper()
            corpus_table = config.DBWPTABLE + "_" + corpus.upper()

            selects.append((corpus.upper(),
                            "(SELECT S1.string AS head, S1.pos AS headpos, F.rel, S2.string AS dep, S2.pos AS deppos, S2.stringextra AS depextra, F.freq, R.freq AS rel_freq, HR.freq AS head_rel_freq, DR.freq AS dep_rel_freq, " + corpus_sql + " AS corpus, F.id " +
                            "FROM `" + corpus_table + "_strings` AS S1, `" + corpus_table + "_strings` AS S2, `" + corpus_table + "` AS F, `" + corpus_table + "_rel` AS R, `" + corpus_table + "_head_rel` AS HR, `" + corpus_table + "_dep_rel` AS DR " +
                            "WHERE S1.string = " + word_sql + " AND F.head = S1.id AND F.wfhead = 1 AND S2.id = F.dep " +
                            minfreqsql +
                            "AND F.rel = R.rel AND F.head = HR.head AND F.rel = HR.rel AND F.dep = DR.dep AND F.rel = DR.rel)"
                            ))
            selects.append((None,
                            "(SELECT S1.string AS head, S1.pos AS headpos, F.rel, S2.string AS dep, S2.pos AS deppos, S2.stringextra AS depextra, F.freq, R.freq AS rel_freq, HR.freq AS head_rel_freq, DR.freq AS dep_rel_freq, " + corpus_sql + " AS corpus, F.id " +
                            "FROM `" + corpus_table + "_strings` AS S1, `" + corpus_table + "_strings` AS S2, `" + corpus_table + "` AS F, `" + corpus_table + "_rel` AS R, `" + corpus_table + "_head_rel` AS HR, `" + corpus_table + "_dep_rel` AS DR " +
                            "WHERE S2.string = " + word_sql + " AND F.dep = S2.id AND F.wfdep = 1 AND S1.id = F.head " +
                            minfreqsql +
                            "AND F.rel = R.rel AND F.head = HR.head AND F.rel = HR.rel AND F.dep = DR.dep AND F.rel = DR.rel)"
                            ))

    cursor_result = []
    if corpora_rest:
        if incremental:
            yield {"progress_corpora": list(corpora_rest)}
            progress_count = 0
            for sql in selects:
                cursor.execute(sql[1])
                cursor_result.extend(list(cursor))
                if sql[0]:
                    yield {"progress_%d" % progress_count: {"corpus": sql[0]}}
                    progress_count += 1
        else:
            sql = " UNION ALL ".join(x[1] for x in selects)
            cursor.execute(sql)
            cursor_result = cursor

    rels = {}
    counter = {}
    freq_rel = {}
    freq_head_rel = {}
    freq_rel_dep = {}

    do_caching = False
    corpus = None
    corpus_data = []
    unique_id = str(uuid.uuid4())

    def save_cache(corpus, data):
        corpus_checksum = get_hash((word, search_type, minfreq))
        cache_filename = os.path.join(config.CACHE_DIR, "%s:relations_%s" % (corpus, corpus_checksum))
        cache_filename_temp = cache_filename + "." + unique_id
        if not os.path.isfile(cache_filename_temp):
            with open(cache_filename_temp, "wb") as f:
                pickle.dump(data, f, protocol=-1)
            os.rename(cache_filename_temp, cache_filename)

    for row in itertools.chain(relations_data, (None,), cursor_result):
        if row is None and args["cache"]:
            do_caching = True
            continue

        if do_caching:
            if corpus is None:
                corpus = row["corpus"]
            elif not row["corpus"] == corpus:
                save_cache(corpus, corpus_data)
                corpus_data = []
                corpus = row["corpus"]
            corpus_data.append(row)

        head = (row["head"], row["headpos"])
        dep = (row["dep"], row["deppos"], row["depextra"])
        rels.setdefault((head, row["rel"], dep), {"freq": 0, "source": set()})
        rels[(head, row["rel"], dep)]["freq"] += row["freq"]
        rels[(head, row["rel"], dep)]["source"].add("%s:%d" % (row["corpus"], row["id"]))
        freq_rel.setdefault(row["rel"], {})[(row["corpus"], row["rel"])] = row["rel_freq"]
        freq_head_rel.setdefault((head, row["rel"]), {})[(row["corpus"], row["rel"])] = row["head_rel_freq"]
        freq_rel_dep.setdefault((row["rel"], dep), {})[(row["corpus"], row["rel"])] = row["dep_rel_freq"]

    if corpus is not None:
        save_cache(corpus, corpus_data)
        del corpus_data

    cursor.close()

    # Calculate MI
    for rel in rels:
        f_rel = sum(freq_rel[rel[1]].values())
        f_head_rel = sum(freq_head_rel[(rel[0], rel[1])].values())
        f_rel_dep = sum(freq_rel_dep[(rel[1], rel[2])].values())
        rels[rel]["mi"] = rels[rel]["freq"] * math.log((f_rel * rels[rel]["freq"]) / (f_head_rel * f_rel_dep * 1.0), 2)

    sortedrels = sorted(rels.items(), key=lambda x: (x[0][1], x[1][sortby]), reverse=True)

    for rel in sortedrels:
        counter.setdefault((rel[0][1], "h"), 0)
        counter.setdefault((rel[0][1], "d"), 0)
        if search_type == "lemgram" and rel[0][0][0] == word:
            counter[(rel[0][1], "h")] += 1
            if maxresults and counter[(rel[0][1], "h")] > maxresults:
                continue
        else:
            counter[(rel[0][1], "d")] += 1
            if maxresults and counter[(rel[0][1], "d")] > maxresults:
                continue

        r = {"head": rel[0][0][0],
             "headpos": rel[0][0][1],
             "rel": rel[0][1],
             "dep": rel[0][2][0],
             "deppos": rel[0][2][1],
             "depextra": rel[0][2][2],
             "freq": rel[1]["freq"],
             "mi": rel[1]["mi"],
             "source": list(rel[1]["source"])
             }
        result.setdefault("relations", []).append(r)

    yield result


################################################################################
# RELATIONS_SENTENCES
################################################################################

@app.route("/relations_sentences", methods=["GET", "POST"])
@main_handler
def relations_sentences(args):
    """Execute a CQP query to find sentences with a given relation from a word picture.

    The required parameters are
     - source: source ID

    The optional parameters are
     - start, end: which result rows that should be returned
     - show
     - show_struct
    """

    assert_key("source", args, "", True)
    assert_key("start", args, IS_NUMBER, False)
    assert_key("end", args, IS_NUMBER, False)

    temp_source = args.get("source")
    if isinstance(temp_source, str):
        temp_source = temp_source.split(QUERY_DELIM)
    source = defaultdict(set)
    for s in temp_source:
        c, i = s.split(":")
        source[c].add(i)

    check_authentication(source.keys())

    start = int(args.get("start") or 0)
    end = int(args.get("end") or 9)
    shown = args.get("show") or "word"
    shown_structs = args.get("show_struct") or []
    if isinstance(shown_structs, str):
        shown_structs = shown_structs.split(QUERY_DELIM)
    shown_structs = set(shown_structs)

    defaultcontext = args.get("defaultcontext") or "1 sentence"

    querystarttime = time.time()

    cursor = mysql.connection.cursor()
    cursor.execute("SET @@session.long_query_time = 1000;")
    selects = []
    counts = []

    # Get available tables
    cursor.execute("SHOW TABLES LIKE '" + config.DBWPTABLE + "_%';")
    tables = set(list(x.values())[0] for x in cursor)
    # Filter out corpora which doesn't exist in database
    source = sorted([x for x in iter(source.items()) if config.DBWPTABLE + "_" + x[0].upper() in tables])
    if not source:
        yield {}
        return
    corpora = [x[0] for x in source]

    for s in source:
        corpus, ids = s
        ids = [int(i) for i in ids]
        ids_list = "(" + ", ".join("%d" % i for i in ids) + ")"

        corpus_table_sentences = config.DBWPTABLE + "_" + corpus.upper() + "_sentences"

        selects.append("(SELECT S.sentence, S.start, S.end, '" + sql_escape(corpus.upper()) + "' AS corpus " +
                       "FROM `" + corpus_table_sentences + "` as S " +
                       " WHERE S.id IN " + ids_list + ")"
                       )
        counts.append("(SELECT '" + sql_escape(corpus.upper()) + "' AS corpus, COUNT(*) AS freq FROM `" +
                      corpus_table_sentences + "` as S WHERE S.id IN " + ids_list + ")")

    sql_count = " UNION ALL ".join(counts)
    cursor.execute(sql_count)

    corpus_hits = {}
    for row in cursor:
        corpus_hits[row["corpus"]] = int(row["freq"])

    sql = " UNION ALL ".join(selects) + (" LIMIT %d, %d" % (start, end - start + 1))
    cursor.execute(sql)

    querytime = time.time() - querystarttime
    corpora_dict = {}
    for row in cursor:
        corpora_dict.setdefault(row["corpus"], {}).setdefault(row["sentence"], []).append((row["start"], row["end"]))

    cursor.close()

    total_hits = sum(corpus_hits.values())

    if not corpora_dict:
        yield {"hits": 0}
        return

    cqpstarttime = time.time()
    result = {}

    for corp, sids in sorted(corpora_dict.items(), key=lambda x: x[0]):
        cqp = u'<sentence_id="%s"> []* </sentence_id> within sentence' % "|".join(set(sids.keys()))
        q = {"cqp": cqp,
             "corpus": corp,
             "start": "0",
             "end": str(end - start),
             "show_struct": ["sentence_id"] + list(shown_structs),
             "defaultcontext": defaultcontext}
        if shown:
            q["show"] = shown
        result_temp = generator_to_dict(query(q))

        # Loop backwards since we might be adding new items
        for i in range(len(result_temp["kwic"]) - 1, -1, -1):
            s = result_temp["kwic"][i]
            sid = s["structs"]["sentence_id"]
            r = sids[sid][0]
            sentence_start = s["match"]["start"]
            s["match"]["start"] = sentence_start + min(map(int, r)) - 1
            s["match"]["end"] = sentence_start + max(map(int, r))

            # If the same relation appears more than once in the same sentence,
            # append copies of the sentence as separate results
            for r in sids[sid][1:]:
                s2 = deepcopy(s)
                s2["match"]["start"] = sentence_start + min(map(int, r)) - 1
                s2["match"]["end"] = sentence_start + max(map(int, r))
                result_temp["kwic"].insert(i + 1, s2)

        result.setdefault("kwic", []).extend(result_temp["kwic"])

    result["hits"] = total_hits
    result["corpus_hits"] = corpus_hits
    result["corpus_order"] = corpora
    result["querytime"] = querytime
    result["cqptime"] = time.time() - cqpstarttime

    yield result


################################################################################
# CACHE HANDLING
################################################################################

@app.route("/cache", methods=["GET", "POST"])
@main_handler
def cache_handler(args):
    if not config.CACHE_DIR:
        return {}

    result = {}

    if "clean" in args:
        now = time.time()

        # Get modification time of corpus registry files
        corpora = dict((os.path.basename(f).upper(), os.path.getmtime(f)) for f in
                       glob.glob(os.path.join(config.CWB_REGISTRY, "*")))
        last_update = max(corpora.values())
        removed = 0
        for corpus in corpora:
            # Remove cache for updated corpora and old query data
            for cachefile in glob.glob(os.path.join(config.CACHE_DIR, "%s:*" % corpus)):
                cachefiletime = os.path.getmtime(cachefile)
                if cachefiletime < corpora[corpus] or \
                        (":query_data_" in cachefile and cachefiletime < (now - 20 * 60)):
                    os.remove(cachefile)
                    removed += 1

        # Remove combined info and timespan caches older than the last updated corpus
        for cachefile in itertools.chain(glob.glob(os.path.join(config.CACHE_DIR, "info*")),
                                         glob.glob(os.path.join(config.CACHE_DIR, "timespan_*"))):
            if os.path.getmtime(cachefile) < last_update:
                os.remove(cachefile)
                removed += 1

        result["removed"] = removed

    yield result


################################################################################
# Helper functions
################################################################################

def parse_cqp(cqp):
    """ Try to parse a CQP query, returning identified tokens and a
    boolean indicating partial failure if True."""

    sections = []
    last_start = 0
    in_bracket = 0
    in_quote = False
    in_curly = False
    quote_type = ""

    for i in range(len(cqp)):
        c = cqp[i]

        if c in '"\'':
            if in_quote and quote_type == c and not cqp[i - 1] == "\\":
                in_quote = False
                if not in_bracket:
                    sections.append([last_start, i])
            elif not in_quote:
                in_quote = True
                quote_type = c
                if not in_bracket:
                    last_start = i
        elif c == "[":
            if not in_bracket and not in_quote:
                last_start = i
                in_bracket = True
                if len(cqp) > i + 1 and cqp[i + 1] == ":":
                    # Zero-width assertion encountered, which can not be handled by MU query
                    return [], True
        elif c == "]":
            if in_bracket and not in_quote:
                sections.append([last_start, i])
                in_bracket = False
        elif c == "{" and not in_bracket and not in_quote:
            in_curly = True
        elif c == "}" and not in_bracket and not in_quote and in_curly:
            in_curly = False
            sections[-1][1] = i

    last_section = (0, 0)
    sections.append([len(cqp), len(cqp)])
    tokens = []
    rest = False

    for section in sections:
        if last_section[1] < section[0]:
            if cqp[last_section[1] + 1:section[0]].strip():
                rest = True
        last_section = section
        if cqp[section[0]:section[1] + 1]:
            tokens.append(cqp[section[0]:section[1] + 1])

    return tokens, rest


def make_cqp(cqp, within=None, cut=None, expand=None):
    """ Combine CQP query and extra options. """
    for arg in (("within", within), ("cut", cut), ("expand", expand)):
        if arg[1]:
            cqp += " %s %s" % arg
    return cqp


def make_query(cqp):
    """Create web-safe commands for a CQP query.
    """
    querylock = random.randrange(10 ** 8, 10 ** 9)
    return ["set QueryLock %s;" % querylock,
            "%s;" % cqp,
            "unlock %s;" % querylock]


def translate_undef(s):
    """Translate None to '__UNDEF__'."""
    return None if s == "__UNDEF__" else s


def get_hash(values):
    """Get a hash for a list of values."""
    return hashlib.sha256(bytes(";".join(v if isinstance(v, str) else str(v) for v in values), "UTF-8")).hexdigest()


class CQPError(Exception):
    pass


class KorpAuthenticationError(Exception):
    pass


class Namespace:
    pass


def run_cqp(command, encoding=None, executable=config.CQP_EXECUTABLE, registry=config.CWB_REGISTRY, attr_ignore=False):
    """Call the CQP binary with the given command, and the request data.
    Yield one result line at the time, disregarding empty lines.
    If there is an error, raise a CQPError exception.
    """
    env = os.environ.copy()
    env["LC_COLLATE"] = config.LC_COLLATE
    encoding = encoding or config.CQP_ENCODING
    if not isinstance(command, str):
        command = "\n".join(command)
    command = "set PrettyPrint off;\n" + command
    command = command.encode(encoding)
    process = subprocess.Popen([executable, "-c", "-r", registry],
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=env)
    reply, error = process.communicate(command)
    if error:
        error = error.decode(encoding)
        # Remove newlines from the error string:
        error = re.sub(r"\s+", r" ", error)
        # Keep only the first CQP error (the rest are consequences):
        error = re.sub(r"^CQP Error: *", r"", error)
        error = re.sub(r" *(CQP Error:).*$", r"", error)
        # Ignore certain errors:
        # 1) "show +attr" for unknown attr,
        # 2) querying unknown structural attribute,
        # 3) calculating statistics for empty results
        if not (attr_ignore and "No such attribute:" in error) \
                and "is not defined for corpus" not in error \
                and "cl->range && cl->size > 0" not in error \
                and "neither a positional/structural attribute" not in error \
                and "CL: major error, cannot compose string: invalid UTF8 string passed to cl_string_canonical..." not in error:
            raise CQPError(error)
    for line in reply.decode(encoding, errors="ignore").split(
            "\n"):  # We don't use splitlines() since it might split on special characters in the data
        if line:
            yield line


def run_cwb_scan(corpus, attrs, encoding=config.CQP_ENCODING, executable=config.CWB_SCAN_EXECUTABLE, registry=config.CWB_REGISTRY):
    """Call the cwb-scan-corpus binary with the given arguments.
    Yield one result line at the time, disregarding empty lines.
    If there is an error, raise a CQPError exception.
    """
    process = subprocess.Popen([executable, "-q", "-r", registry, corpus] + attrs,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    reply, error = process.communicate()
    if error:
        # Remove newlines from the error string:
        error = re.sub(r"\s+", r" ", error.decode())
        # Ignore certain errors:
        # 1) "show +attr" for unknown attr,
        # 2) querying unknown structural attribute,
        # 3) calculating statistics for empty results
        raise CQPError(error)
    for line in reply.decode(encoding, errors="ignore").split(
            "\n"):  # We don't use splitlines() since it might split on special characters in the data
        if line and len(line) < 65536:
            yield line


def show_attributes():
    """Command sequence for returning the corpus attributes."""
    return ["show cd; .EOL.;"]


def read_attributes(lines):
    """Read the CQP output from the show_attributes() command."""
    attrs = {'p': [], 's': [], 'a': []}
    for line in lines:
        if line == END_OF_LINE:
            break
        (typ, name, _rest) = (line + " X").split(None, 2)
        attrs[typ[0]].append(name)
    return attrs


def assert_key(key, form, regexp, required=False):
    """Check that the value of the attribute 'key' in the request data
    matches the specification 'regexp'. If 'required' is True, then
    the key has to be in the form.
    """
    value = form.get(key, "")
    if value and not isinstance(value, list):
        value = [value]
    if required and not value:
        raise KeyError("Key is required: %s" % key)
    if not all(re.match(regexp, x) for x in value):
        pattern = regexp.pattern if hasattr(regexp, "pattern") else regexp
        raise ValueError("Value(s) for key %s do(es) not match /%s/: %s" % (key, pattern, value))


@app.route("/authenticate", methods=["GET", "POST"])
@main_handler
def authenticate(_=None):
    """Authenticate a user against an authentication server.
    """

    auth_data = request.authorization

    if auth_data:
        postdata = {
            "username": auth_data["username"],
            "password": auth_data["password"],
            "checksum": hashlib.md5(bytes(auth_data["username"] + auth_data["password"] +
                                          config.AUTH_SECRET, "utf-8")).hexdigest()
        }

        try:
            contents = urllib.request.urlopen(config.AUTH_SERVER,
                                              urllib.parse.urlencode(postdata).encode("utf-8")).read().decode("utf-8")
            auth_response = json.loads(contents)
        except urllib.error.HTTPError:
            raise KorpAuthenticationError("Could not contact authentication server.")
        except ValueError:
            raise KorpAuthenticationError("Invalid response from authentication server.")
        except:
            raise KorpAuthenticationError("Unexpected error during authentication.")

        if auth_response["authenticated"]:
            permitted_resources = auth_response["permitted_resources"]
            result = {"corpora": []}
            if "corpora" in permitted_resources:
                for c in permitted_resources["corpora"]:
                    if permitted_resources["corpora"][c]["read"]:
                        result["corpora"].append(c.upper())
            yield result
            return

    yield {}


def check_authentication(corpora):
    """Take a list of corpora, and if any of them are protected, run authentication.
    Raises an error if authentication fails."""

    if config.PROTECTED_FILE:
        # Split parallel corpora
        corpora = [cc for c in corpora for cc in c.split("|")]
        with open(config.PROTECTED_FILE) as infile:
            protected = [x.strip() for x in infile.readlines()]
        c = [c for c in corpora if c.upper() in protected]
        if c:
            auth = generator_to_dict(authenticate({}))
            unauthorized = [x for x in c if x.upper() not in auth.get("corpora", [])]
            if not auth or unauthorized:
                raise KorpAuthenticationError("You do not have access to the following corpora: %s" %
                                              ", ".join(unauthorized))


def generator_to_dict(generator):
    d = next(generator)
    for v in generator:
        d.update(v)
    return d


def reraise_with_stack(func):
    """ Wrapper to preserve traceback when using ThreadPoolExecutor. """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Raise an exception of the same type with the traceback as message
            raise sys.exc_info()[0](traceback.format_exc())

    return wrapped


class CustomTracebackException(Exception):
    def __init__(self, exception):
        self.exception = exception


def prevent_timeout(f, args=None, timeout=15):
    """ Used in places where the script otherwise might timeout. Keeps the connection alive by printing
    out whitespace. """

    q = Queue()

    def error_catcher(g, *args, **kwargs):
        try:
            g(*args, **kwargs)
        except Exception as e:
            q.put(sys.exc_info())

    args = args or []
    args.append(q)

    pool = ThreadPool(1)
    pool.spawn(error_catcher, f, *args)

    while True:
        try:
            msg = q.get(block=True, timeout=timeout)
            if msg == "DONE":
                break
            elif isinstance(msg, tuple):
                raise CustomTracebackException(msg)
            else:
                yield msg
        except Empty:
            yield {}


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "dev":
        # Run using Flask (use only for development)
        app.run(debug=True, threaded=True, host=config.WSGI_HOST, port=config.WSGI_PORT)
    else:
        # Run using gevent
        print("Serving using gevent")
        http = WSGIServer((config.WSGI_HOST, config.WSGI_PORT), app.wsgi_app)
        http.serve_forever()

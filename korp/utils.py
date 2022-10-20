import datetime
import functools
import glob
import hashlib
import importlib
import json
import os
import random
import re
import sys
import time
import traceback
from collections import defaultdict
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

from flask import Response, request, copy_current_request_context, stream_with_context
from flask import current_app as app
from flask.blueprints import Blueprint
from gevent.queue import Queue, Empty
from gevent.threadpool import ThreadPool

from korp.db import mysql
from korp.memcached import memcached

# Special symbols used by this script; they must NOT be in the corpus
END_OF_LINE = "-::-EOL-::-"
LEFT_DELIM = "---:::"
RIGHT_DELIM = ":::---"

# Regular expressions for parsing parameters
IS_NUMBER = re.compile(r"^\d+$")
IS_IDENT = re.compile(r"^[\w\-,|]+$")

QUERY_DELIM = ","

authorizer: Optional["Authorizer"] = None


def main_handler(generator):
    """Decorator wrapping all WSGI endpoints, handling errors and formatting.

    Global parameters are
     - callback: an identifier that the result should be wrapped in
     - encoding: the encoding for interacting with the corpus (default: UTF-8)
     - indent: pretty-print the result with a specific indentation
     - debug: if set, return some extra information (for debugging)
    """
    @functools.wraps(generator)  # Copy original function's information, needed by Flask
    def decorated(args=None, *pargs, **kwargs):
        internal = args is not None
        if not internal:
            if request.is_json:
                args = request.get_json()
            else:
                args = request.values.to_dict()

        args["internal"] = internal

        if not isinstance(args.get("cache"), bool):
            args["cache"] = bool(not app.config["CACHE_DISABLED"] and
                                 not args.get("cache", "").lower() == "false" and
                                 app.config["CACHE_DIR"] and os.path.exists(app.config["CACHE_DIR"]) and
                                 app.config["MEMCACHED_SERVERS"])

        if internal:
            # Function is internally used
            return generator(args, *pargs, **kwargs)
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
                """Yield full JSON at the end, but until then keep returning newlines to prevent timeout."""
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
            incremental = parse_bool(args, "incremental", False)
            callback = args.get("callback")
            indent = int(args.get("indent", 0))

            if incremental:
                # Incremental response
                return Response(stream_with_context(incremental_json(generator(args, *pargs, **kwargs))),
                                mimetype="application/json")
            else:
                # We still use a streaming response even when non-incremental, to prevent timeouts
                return Response(stream_with_context(full_json(generator(args, *pargs, **kwargs))),
                                mimetype="application/json")

    return decorated


def prevent_timeout(generator):
    """Decorator for long-running functions that might otherwise timeout."""
    @functools.wraps(generator)
    def decorated(args=None, *pargs, **kwargs):
        if args["internal"]:
            # Internally used
            yield from generator(args, *pargs, **kwargs)
            return

        def f(queue):
            for response in generator(args, *pargs, **kwargs):
                queue.put(response)
            queue.put("DONE")

        timeout = 15
        q = Queue()

        @copy_current_request_context
        def error_catcher(g, *pargs, **kwargs):
            try:
                g(*pargs, **kwargs)
            except Exception:
                q.put(sys.exc_info())

        pool = ThreadPool(1)
        pool.spawn(error_catcher, f, q)

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

    return decorated


def generator_to_dict(generator):
    d = next(generator)
    for v in generator:
        d.update(v)
    return d


def parse_bool(args, key, default=True):
    if default:
        return args.get(key, "").lower() != "false"
    else:
        return args.get(key, "").lower() == "true"


class CustomTracebackException(Exception):
    def __init__(self, exception):
        self.exception = exception


def get_corpus_timestamps():
    """Get modification time of corpus registry files."""
    corpora = dict((os.path.basename(f).upper(), os.path.getmtime(f)) for f in
                   glob.glob(os.path.join(app.config["CWB_REGISTRY"], "*")))
    return corpora


def get_corpus_config_timestamps():
    """Get modification time of corpus config files."""
    corpora = dict((os.path.basename(f)[:-5].upper(), os.path.getmtime(f)) for f in
                   glob.glob(os.path.join(app.config["CORPUS_CONFIG_DIR"], "corpora", "*.yaml")))
    modes = max(os.path.getmtime(f) for f in glob.glob(os.path.join(app.config["CORPUS_CONFIG_DIR"],
                                                                    "modes", "*.yaml")))
    presets = max(
        os.path.getmtime(f) for f in glob.glob(os.path.join(app.config["CORPUS_CONFIG_DIR"], "attributes", "*/*.yaml")))
    return corpora, modes, presets


def setup_cache():
    """Setup disk cache and Memcached if needed."""
    action_needed = False

    # Create cache dir if needed
    if app.config["CACHE_DIR"] and not os.path.exists(app.config["CACHE_DIR"]):
        os.makedirs(app.config["CACHE_DIR"])
        action_needed = True

    # Set up Memcached if needed
    if app.config["MEMCACHED_SERVERS"]:
        with memcached.pool.reserve() as mc:
            if "multi:version" not in mc:
                corpora = get_corpus_timestamps()
                corpora_configs, config_modes, config_presets = get_corpus_config_timestamps()
                mc.set("multi:version", 1)
                mc.set("multi:version_config", 1)
                mc.set("multi:corpora", set(corpora.keys()))
                mc.set("multi:config_corpora", set(corpora_configs.keys()))
                mc.set("multi:config_modes", config_modes)
                mc.set("multi:config_presets", config_presets)
                for corpus in corpora:
                    mc.set("%s:version" % corpus, 1)
                    mc.set("%s:version_config" % corpus, 1)
                    mc.set("%s:last_update" % corpus, corpora[corpus])
                    mc.set("%s:last_update_config" % corpus, corpora_configs.get(corpus, 0))
                action_needed = True

    return action_needed


def cache_prefix(mc, corpus="multi", config=False):
    """Get cache version to use as prefix for cache keys."""

    return "%s:%d" % (corpus, mc.get(f"{corpus}:version{'_config' if config else ''}", 0))


def query_optimize(cqp, cqpparams, find_match=True, expand=True, free_search=False):
    """Optimize simple queries with multiple words by converting them to MU queries.
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
        # Most of the time we only need to expand to the right, except for when leading wildcards are used.
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


################################################################################
# ARGUMENT PARSING
################################################################################

def parse_corpora(args):
    corpora = args.get("corpus", [])
    if isinstance(corpora, str):
        corpora = corpora.upper().split(QUERY_DELIM)
    return sorted(set(corpora))


def parse_within(args):
    within = defaultdict(lambda: args.get("default_within"))

    if args.get("within"):
        if ":" not in args.get("within"):
            raise ValueError("Malformed value for key 'within'.")
        within.update({x.split(":")[0].upper(): x.split(":")[1] for x in args.get("within").split(QUERY_DELIM)})
    return within


def parse_cqp_subcqp(args):
    cqp = [args.get(key) for key in sorted([k for k in args.keys() if k.startswith("cqp")],
                                           key=lambda x: int(x[3:]) if len(x) > 3 else 0)]
    subcqp = [args.get(key) for key in sorted([k for k in args.keys() if k.startswith("subcqp")],
                                              key=lambda x: int(x[6:]) if len(x) > 6 else 0)]
    return cqp, subcqp


################################################################################
# Helper functions
################################################################################

def parse_cqp(cqp):
    """Try to parse a CQP query, returning identified tokens and a
    boolean indicating partial failure if True.
    """
    sections = []
    last_start = 0
    in_bracket = 0
    in_quote = False
    in_curly = False
    escaping = False
    quote_type = ""

    for i in range(len(cqp)):
        c = cqp[i]

        if in_quote and not escaping and c == "\\":
            # Next character is being escaped
            escaping = True
        elif escaping:
            # Current character is being escaped
            escaping = False
        elif c in '"\'':
            if in_quote and quote_type == c:
                if i < len(cqp) - 1 and cqp[i + 1] == quote_type:
                    # First character of a quote escaped by doubling
                    escaping = True
                else:
                    # End of a quote
                    in_quote = False
                    if not in_bracket:
                        sections.append([last_start, i])
            elif not in_quote:
                # Beginning of a qoute
                in_quote = True
                quote_type = c
                if not in_bracket:
                    last_start = i
        elif c == "[":
            if not in_bracket and not in_quote:
                # Beginning of a token
                last_start = i
                in_bracket = True
                if len(cqp) > i + 1 and cqp[i + 1] == ":":
                    # Zero-width assertion encountered, which can not be handled by MU query
                    return [], True
        elif c == "]":
            if in_bracket and not in_quote:
                # End of a token
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
    """Translate '__UNDEF__' to None."""
    return None if s == "__UNDEF__" else s


def get_hash(values):
    """Get a hash for a list of values."""
    return hashlib.sha256(bytes(";".join(v if isinstance(v, str) else str(v) for v in values), "UTF-8")).hexdigest()


class CQPError(Exception):
    pass


class KorpAuthorizationError(Exception):
    pass


class Namespace:
    pass


def assert_key(key, attrs, regexp, required=False):
    """Check that the value of the attribute 'key' in the request data
    matches the specification 'regexp'. If 'required' is True, then
    the key has to be in the form.
    """
    value = None
    if isinstance(key, (tuple, list)):
        for k in key:
            value = attrs.get(k)
            if value is not None:
                break
    else:
        value = attrs.get(key, "")
        key = (key,)
    if value and not isinstance(value, list):
        value = [value]
    if required and not value:
        raise KeyError("Key is required: <%s>" % "|".join(key))
    if value and not all(re.match(regexp, x) for x in value):
        pattern = regexp.pattern if hasattr(regexp, "pattern") else regexp
        raise ValueError("Value(s) for key <%s> do(es) not match /%s/: %s" % ("|".join(key), pattern, value))


def get_protected_corpora() -> List[str]:
    """Return a list of corpora with restricted access."""
    if authorizer:
        return authorizer.get_protected_corpora()
    else:
        return []


def check_authorization(corpora) -> None:
    """Take a list of corpora, and if any of them are protected, check authorization.
    Raises an error if authorization fails."""

    if authorizer:
        # Split parallel corpora
        corpora = [cc for c in corpora for cc in c.split("|")]

        success, unauthorized, message = authorizer.check_authorization(corpora)
        if not success:
            if not message:
                message = "You do not have access to the following corpora: %s" % ", ".join(unauthorized)
            raise KorpAuthorizationError(message)


def strptime(date):
    """Take a date in string format and return a datetime object.
    Input must be on the format "YYYYMMDDhhmmss".
    We need this since the built-in strptime isn't thread safe (and this is much faster)."""
    year = int(date[:4])
    month = int(date[4:6]) if len(date) > 4 else 1
    day = int(date[6:8]) if len(date) > 6 else 1
    hour = int(date[8:10]) if len(date) > 8 else 0
    minute = int(date[10:12]) if len(date) > 10 else 0
    second = int(date[12:14]) if len(date) > 12 else 0
    return datetime.datetime(year, month, day, hour, minute, second)


def sql_escape(s):
    with app.app_context():
        return mysql.connection.escape_string(s).decode("utf-8") if isinstance(s, str) else s


class Plugin(Blueprint):
    """Simple plugin class, identical to Flask's Blueprint but with a method for accessing the plugin's
    configuration."""

    def config(self, key, default=None):
        return app.config["PLUGINS_CONFIG"].get(self.import_name, {}).get(key, default)


class Authorizer(ABC):
    """Class to subclass when implementing an authorizer plugin."""

    auth_class = None

    def __init__(self):
        pass

    def __init_subclass__(cls):
        Authorizer.auth_class = cls

    @abstractmethod
    def get_protected_corpora(self, use_cache: bool = True) -> List[str]:
        """Get list of corpora with restricted access, in uppercase."""
        pass

    @abstractmethod
    def check_authorization(self, corpora: List[str]) -> Tuple[bool, List[str], Optional[str]]:
        """Take a list of corpora and check that the user has permission to access them."""
        pass

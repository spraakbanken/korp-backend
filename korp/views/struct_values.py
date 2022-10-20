import itertools
import re
from collections import defaultdict
from copy import deepcopy
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor

try:
    import pylibmc
except ImportError:
    pylibmc = None
from flask import Blueprint
from flask import current_app as app

from korp import utils
from korp.memcached import memcached
from . import count

bp = Blueprint("struct_values", __name__)


@bp.route("/struct_values", methods=["GET", "POST"])
@utils.main_handler
@utils.prevent_timeout
def struct_values(args):
    """Get all available values for one or more structural attributes."""
    utils.assert_key("corpus", args, utils.IS_IDENT, True)
    utils.assert_key("struct", args, re.compile(r"^[\w_\d,>]+$"), True)
    utils.assert_key("incremental", args, r"(true|false)")

    incremental = utils.parse_bool(args, "incremental", False)
    include_count = utils.parse_bool(args, "count", False)

    per_corpus = utils.parse_bool(args, "per_corpus", True)
    combined = utils.parse_bool(args, "combined", True)
    corpora = utils.parse_corpora(args)
    utils.check_authorization(corpora)

    structs = args.get("struct")
    if isinstance(structs, str):
        structs = structs.split(utils.QUERY_DELIM)

    split = args.get("split", "")
    if isinstance(split, str):
        split = split.split(utils.QUERY_DELIM)

    ns = utils.Namespace()  # To make variables writable from nested functions

    result = {"corpora": defaultdict(dict), "combined": {}}

    from_cache = set()  # Keep track of what has been read from cache

    if args["cache"]:
        all_cache = True
        for corpus in corpora:
            for struct in structs:
                checksum = utils.get_hash((corpus, struct, split, include_count))
                with memcached.pool.reserve() as mc:
                    data = mc.get("%s:struct_values_%s" % (utils.cache_prefix(mc, corpus), checksum))
                if data is not None:
                    result["corpora"].setdefault(corpus, {})
                    result["corpora"][corpus][struct] = data
                    if "debug" in args:
                        result.setdefault("DEBUG", {"caches_read": []})
                        result["DEBUG"]["caches_read"].append("%s:%s" % (corpus, struct))
                    from_cache.add((corpus, struct))
                else:
                    all_cache = False
    else:
        all_cache = False

    if not all_cache:
        ns.progress_count = 0
        if incremental:
            yield {"progress_corpora": list(corpora)}

        with ThreadPoolExecutor(max_workers=app.config["PARALLEL_THREADS"]) as executor:
            future_query = dict((executor.submit(count.count_query_worker_simple, corpus, cqp=None,
                                                 group_by=[(s, True) for s in struct.split(">")],
                                                 use_cache=args["cache"]), (corpus, struct))
                                for corpus in corpora for struct in structs if not (corpus, struct) in from_cache)

            for future in futures.as_completed(future_query):
                corpus, struct = future_query[future]
                if future.exception() is not None:
                    raise utils.CQPError(future.exception())
                else:
                    lines, nr_hits, corpus_size = future.result()

                    corpus_stats = {} if include_count else set()
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
                                    if include_count and i == len(val) - 1:
                                        prev.setdefault(n, 0)
                                        prev[n] += int(freq)
                                        break
                                    elif not include_count and i == len(val) - 1:
                                        prev.append(n)
                                        break
                                    elif not include_count and i == len(val) - 2:
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
                                if include_count:
                                    corpus_stats[val] = int(freq)
                                else:
                                    corpus_stats.add(val)

                    if ">" in struct:
                        result["corpora"][corpus][struct] = vals_dict
                    elif corpus_stats:
                        result["corpora"][corpus][struct] = corpus_stats if include_count else sorted(corpus_stats)

                    if incremental:
                        yield {"progress_%d" % ns.progress_count: corpus}
                        ns.progress_count += 1

    def merge(d1, d2):
        merged = deepcopy(d1)
        for key in d2:
            if key in d1:
                if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    merged[key] = merge(d1[key], d2[key])
                elif isinstance(d1[key], int):
                    merged[key] += d2[key]
                elif isinstance(d1[key], list):
                    merged[key].extend(d2[key])
                    merged[key] = sorted(set(merged[key]))
            else:
                merged[key] = d2[key]
        return merged

    if combined:
        for corpus in result["corpora"]:
            result["combined"] = merge(result["combined"], result["corpora"][corpus])
    else:
        del result["combined"]

    if args["cache"] and not all_cache:
        for corpus in corpora:
            for struct in structs:
                if (corpus, struct) in from_cache:
                    continue
                checksum = utils.get_hash((corpus, struct, split, include_count))
                try:
                    with memcached.pool.reserve() as mc:
                        cache_key = "%s:struct_values_%s" % (utils.cache_prefix(mc, corpus), checksum)
                        mc.add(cache_key, result["corpora"][corpus].get(struct, {}))
                except pylibmc.TooBig:
                    pass
                else:
                    if "debug" in args:
                        result.setdefault("DEBUG", {})
                        result["DEBUG"].setdefault("caches_saved", [])
                        result["DEBUG"]["caches_saved"].append("%s:%s" % (corpus, struct))

    if not per_corpus:
        del result["corpora"]

    yield result

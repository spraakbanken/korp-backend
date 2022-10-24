import itertools
from collections import defaultdict
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor

from dateutil.relativedelta import relativedelta
from flask import Blueprint
from flask import current_app as app
from pymemcache.exceptions import MemcacheError

from korp import utils
from korp.cwb import cwb
from korp.memcached import memcached
from . import info, timespan

bp = Blueprint("count", __name__)


@bp.route("/count", methods=["GET", "POST"])
@utils.main_handler
@utils.prevent_timeout
def count(args):
    """Perform a CQP query and return a count of the given words/attributes."""
    utils.assert_key("cqp", args, r"", True)
    utils.assert_key("corpus", args, utils.IS_IDENT, True)
    utils.assert_key("group_by", args, utils.IS_IDENT, False)
    utils.assert_key("group_by_struct", args, utils.IS_IDENT, False)
    utils.assert_key("cut", args, utils.IS_NUMBER)
    utils.assert_key("ignore_case", args, utils.IS_IDENT)
    utils.assert_key("incremental", args, r"(true|false)")

    incremental = utils.parse_bool(args, "incremental", False)

    corpora = utils.parse_corpora(args)
    utils.check_authorization(corpora)

    group_by = args.get("group_by") or []
    if isinstance(group_by, str):
        group_by = sorted(set(group_by.split(utils.QUERY_DELIM)))

    group_by_struct = args.get("group_by_struct") or []
    if isinstance(group_by_struct, str):
        group_by_struct = sorted(set(group_by_struct.split(utils.QUERY_DELIM)))

    if not group_by and not group_by_struct:
        group_by = ["word"]

    group_by = [(g, False) for g in group_by] + [(g, True) for g in group_by_struct]

    ignore_case = args.get("ignore_case") or []
    if isinstance(ignore_case, str):
        ignore_case = ignore_case.split(utils.QUERY_DELIM)
    ignore_case = set(ignore_case)

    within = utils.parse_within(args)

    relative_to_struct = args.get("relative_to_struct") or []
    if isinstance(relative_to_struct, str):
        relative_to_struct = sorted(set(relative_to_struct.split(utils.QUERY_DELIM)))
    assert all(r in group_by_struct for r in
               relative_to_struct), "All 'relative_to_struct' values also need to be present in 'group_by_struct'."

    relative_to = [(r, True) for r in relative_to_struct]

    start = int(args.get("start") or 0)
    end = int(args.get("end") or -1)

    split = args.get("split") or []
    if isinstance(split, str):
        split = split.split(utils.QUERY_DELIM)

    strip_pointer = args.get("strip_pointer", "")
    if isinstance(strip_pointer, str):
        strip_pointer = strip_pointer.split(utils.QUERY_DELIM)

    top = args.get("top", "")
    if isinstance(top, str):
        if ":" in top:
            top = dict((x.split(":")[0], int(x.split(":")[1])) for x in top.split(utils.QUERY_DELIM))
        else:
            top = dict((x, 1) for x in top.split(utils.QUERY_DELIM))

    expand_prequeries = utils.parse_bool(args, "expand_prequeries", True)

    # Sort numbered CQP-queries numerically
    cqp, subcqp = utils.parse_cqp_subcqp(args)

    if len(cqp) > 1 and expand_prequeries and not all(within[c] for c in corpora):
        raise ValueError("Multiple CQP queries requires 'within' or 'expand_prequeries=false'")

    if subcqp:
        cqp.append(subcqp)

    simple = utils.parse_bool(args, "simple", False)

    if cqp == ["[]"]:
        simple = True

    result = {"corpora": {}}
    debug = {}
    zero_hits = []
    read_from_cache = 0

    if args["cache"]:
        # Use cache to skip corpora with zero hits
        memcached_keys = {}
        with memcached.get_client() as mc:
            cache_prefixes = utils.cache_prefix(mc, corpora)
            for corpus in corpora:
                corpus_checksum = utils.get_hash((cqp,
                                                 group_by,
                                                 within[corpus],
                                                 sorted(ignore_case),
                                                 expand_prequeries))
                memcached_keys["%s:count_size_%s" % (cache_prefixes[corpus], corpus_checksum)] = corpus

            cached_size = mc.get_many(memcached_keys.keys())
        for key in cached_size:
            nr_hits = cached_size[key][0]
            read_from_cache += 1
            if nr_hits == 0:
                zero_hits.append(memcached_keys[key])

        if "debug" in args:
            debug["cache_coverage"] = "%d/%d" % (read_from_cache, len(corpora))

    total_stats = [{"rows": defaultdict(lambda: {"absolute": 0, "relative": 0.0}),
                    "sums": {"absolute": 0, "relative": 0.0}} for _ in range(len(subcqp) + 1)]

    ns = utils.Namespace()  # To make variables writable from nested functions
    ns.total_size = 0

    if relative_to:
        relative_args = {
            "cqp": "[]",
            "corpus": args.get("corpus"),
            "group_by_struct": relative_to_struct,
            "split": split
        }

        relative_to_result = utils.generator_to_dict(count(relative_args))
        relative_to_freqs = {"combined": {}, "corpora": defaultdict(dict)}

        for row in relative_to_result["combined"]["rows"]:
            relative_to_freqs["combined"][tuple(v for k, v in sorted(row["value"].items()))] = row["absolute"]

        for corpus in relative_to_result["corpora"]:
            for row in relative_to_result["corpora"][corpus]["rows"]:
                relative_to_freqs["corpora"][corpus][tuple(v for k, v in sorted(row["value"].items()))] = row[
                    "absolute"]

    count_function = count_query_worker if not simple else count_query_worker_simple

    ns.progress_count = 0
    if incremental:
        yield {"progress_corpora": list(c for c in corpora if c not in zero_hits)}

    for corpus in zero_hits:
        result["corpora"][corpus] = [{"rows": {},
                                      "sums": {"absolute": 0, "relative": 0.0}} for _ in range(len(subcqp) + 1)]
        for i in range(len(subcqp)):
            result["corpora"][corpus][i + 1]["cqp"] = subcqp[i]

    with ThreadPoolExecutor(max_workers=app.config["PARALLEL_THREADS"]) as executor:
        future_query = dict((executor.submit(count_function, corpus=corpus, cqp=cqp, group_by=group_by,
                                             within=within[corpus], ignore_case=ignore_case,
                                             expand_prequeries=expand_prequeries,
                                             use_cache=args["cache"], cache_max=app.config["CACHE_MAX_STATS"]), corpus)
                            for corpus in corpora if corpus not in zero_hits)

        for future in futures.as_completed(future_query):
            corpus = future_query[future]
            if future.exception() is not None:
                raise utils.CQPError(future.exception())
            else:
                lines, nr_hits, corpus_size = future.result()

                ns.total_size += corpus_size
                corpus_stats = [{"rows": defaultdict(lambda: {"absolute": 0, "relative": 0.0}),
                                 "sums": {"absolute": 0, "relative": 0.0}} for _ in range(len(subcqp) + 1)]

                query_no = 0
                for line in lines:
                    if line == utils.END_OF_LINE:
                        # EOL means the start of a new subcqp result
                        query_no += 1
                        if subcqp:
                            corpus_stats[query_no]["cqp"] = subcqp[query_no - 1]
                        continue
                    freq, ngram = line.lstrip().split(" ", 1)

                    if len(group_by) > 1:
                        ngram_groups = ngram.split("\t")
                    else:
                        ngram_groups = [ngram]

                    all_ngrams = []
                    relative_to_pos = []

                    for i, ngram in enumerate(ngram_groups):
                        # Split value sets and treat each value as a hit
                        if group_by[i][0] in split:
                            tokens = [t + "|" for t in ngram.split(
                                "| ")]  # We can't split on just space due to spaces in annotations
                            tokens[-1] = tokens[-1][:-1]
                            if group_by[i][0] in top:
                                split_tokens = [[x for x in token.split("|") if x][:top[group_by[i][0]]]
                                                if not token == "|" else ["|"] for token in tokens]
                            else:
                                split_tokens = [[x for x in token.split("|") if x] if not token == "|" else [""]
                                                for token in tokens]
                            ngrams = itertools.product(*split_tokens)
                            ngrams = tuple(x for x in ngrams)
                        else:
                            if not group_by[i][1]:
                                ngrams = (tuple(ngram.split(" ")),)
                            else:
                                ngrams = (ngram,)

                        # Remove multi-word pointers
                        if group_by[i][0] in strip_pointer:
                            for j in range(len(ngrams)):
                                for k in range(len(ngrams[j])):
                                    if ":" in ngrams[j][k]:
                                        ngramtemp, pointer = ngrams[j][k].rsplit(":", 1)
                                        if pointer.isnumeric():
                                            ngrams[j][k] = ngramtemp

                        all_ngrams.append(ngrams)

                        if relative_to and group_by[i] in relative_to:
                            relative_to_pos.append(i)

                    cross = list(itertools.product(*all_ngrams))

                    for ngram in cross:
                        corpus_stats[query_no]["rows"][ngram]["absolute"] += int(freq)
                        corpus_stats[query_no]["sums"]["absolute"] += int(freq)
                        total_stats[query_no]["rows"][ngram]["absolute"] += int(freq)
                        total_stats[query_no]["sums"]["absolute"] += int(freq)

                        if relative_to:
                            relativeto_ngram = tuple(ngram[pos] for pos in relative_to_pos)
                            corpus_stats[query_no]["rows"][ngram]["relative"] += int(freq) / float(
                                relative_to_freqs["corpora"][corpus][relativeto_ngram]) * 1000000
                            corpus_stats[query_no]["sums"]["relative"] += int(freq) / float(
                                relative_to_freqs["corpora"][corpus][relativeto_ngram]) * 1000000
                            total_stats[query_no]["rows"][ngram]["relative"] += int(freq) / float(
                                relative_to_freqs["combined"][relativeto_ngram]) * 1000000
                        else:
                            corpus_stats[query_no]["rows"][ngram]["relative"] += int(freq) / float(
                                corpus_size) * 1000000
                            corpus_stats[query_no]["sums"]["relative"] += int(freq) / float(corpus_size) * 1000000

                result["corpora"][corpus] = corpus_stats

                if incremental:
                    yield {"progress_%d" % ns.progress_count: corpus}
                    ns.progress_count += 1

    result["count"] = len(total_stats[0]["rows"])

    # Calculate relative numbers for the total
    for query_no in range(len(subcqp) + 1):
        if end > -1 and (start > 0 or len(total_stats[0]["rows"]) > (end - start) + 1):
            # Only a selected range of results requested
            total_stats[query_no]["rows"] = dict(
                sorted(total_stats[query_no]["rows"].items(), key=lambda x: x[1]["absolute"],
                       reverse=True)[start:end + 1])

            for corpus in corpora:
                result["corpora"][corpus][query_no]["rows"] = {k: v for k, v in result["corpora"][corpus][query_no][
                    "rows"].items() if k in total_stats[query_no]["rows"]}

        if not relative_to:
            for ngram, vals in total_stats[query_no]["rows"].items():
                total_stats[query_no]["rows"][ngram]["relative"] = vals["absolute"] / float(ns.total_size) * 1000000

        for corpus in corpora:
            new_list = []
            for ngram, vals in result["corpora"][corpus][query_no]["rows"].items():
                row = {"value": {key[0]: ngram[i] for i, key in enumerate(group_by)}}
                row.update(vals)
                new_list.append(row)
            result["corpora"][corpus][query_no]["rows"] = new_list

        total_stats[query_no]["sums"]["relative"] = (total_stats[query_no]["sums"]["absolute"] / float(ns.total_size)
                                                     * 1000000 if ns.total_size > 0 else 0.0)

        if subcqp and query_no > 0:
            total_stats[query_no]["cqp"] = subcqp[query_no - 1]

        new_list = []
        for ngram, vals in total_stats[query_no]["rows"].items():
            row = {"value": dict((key[0], ngram[i]) for i, key in enumerate(group_by))}
            row.update(vals)
            new_list.append(row)
        total_stats[query_no]["rows"] = new_list

    result["combined"] = total_stats if len(total_stats) > 1 else total_stats[0]

    if not subcqp:
        for corpus in corpora:
            result["corpora"][corpus] = result["corpora"][corpus][0]

    if "debug" in args:
        debug.update({"cqp": cqp, "simple": simple})
        result["DEBUG"] = debug

    yield result


@bp.route("/count_all", methods=["GET", "POST"])
@utils.main_handler
@utils.prevent_timeout
def count_all(args):
    """Like /count but for every single value of the given attributes."""
    utils.assert_key("corpus", args, utils.IS_IDENT, True)
    utils.assert_key(("group_by", "group_by_struct"), args, utils.IS_IDENT, True)
    utils.assert_key("cut", args, utils.IS_NUMBER)
    utils.assert_key("ignore_case", args, utils.IS_IDENT)
    utils.assert_key("incremental", args, r"(true|false)")

    args["cqp"] = "[]"  # Dummy value, not used
    args["simple"] = "true"

    yield utils.generator_to_dict(count(args))


@bp.route("/count_time", methods=["GET", "POST"])
@utils.main_handler
@utils.prevent_timeout
def count_time(args):
    """Count occurrences per time period."""
    utils.assert_key("cqp", args, r"", True)
    utils.assert_key("corpus", args, utils.IS_IDENT, True)
    utils.assert_key("cut", args, utils.IS_NUMBER)
    utils.assert_key("incremental", args, r"(true|false)")
    utils.assert_key("granularity", args, r"[ymdhnsYMDHNS]")
    utils.assert_key("from", args, r"^\d{14}$")
    utils.assert_key("to", args, r"^\d{14}$")
    utils.assert_key("strategy", args, r"^[123]$")
    utils.assert_key("combined", args, r"(true|false)")
    utils.assert_key("per_corpus", args, r"(true|false)")

    incremental = utils.parse_bool(args, "incremental", False)
    combined = utils.parse_bool(args, "combined", True)
    per_corpus = utils.parse_bool(args, "per_corpus", True)

    corpora = utils.parse_corpora(args)
    utils.check_authorization(corpora)
    within = utils.parse_within(args)
    expand_prequeries = utils.parse_bool(args, "expand_prequeries", True)

    # Sort numbered CQP-queries numerically
    cqp, subcqp = utils.parse_cqp_subcqp(args)

    if len(cqp) > 1 and expand_prequeries and not all(within[c] for c in corpora):
        raise ValueError("Multiple CQP queries requires 'within' or 'expand_prequeries=false'")

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

    result = {}
    if per_corpus:
        result["corpora"] = {}
    if "debug" in args:
        result["DEBUG"] = {"cqp": cqp}

    # Get date range of selected corpora
    corpus_data = utils.generator_to_dict(
        info.corpus_info({"corpus": utils.QUERY_DELIM.join(corpora), "cache": args["cache"]}, no_combined_cache=True))
    corpora_copy = corpora.copy()

    if fromdate and todate:
        df = utils.strptime(fromdate)
        dt = utils.strptime(todate)

        # Remove corpora not within selected date span
        for c in corpus_data["corpora"]:
            firstdate = corpus_data["corpora"][c]["info"].get("FirstDate")
            lastdate = corpus_data["corpora"][c]["info"].get("LastDate")
            if firstdate and lastdate:
                firstdate = utils.strptime(firstdate.replace("-", "").replace(":", "").replace(" ", ""))
                lastdate = utils.strptime(lastdate.replace("-", "").replace(":", "").replace(" ", ""))

                if not (firstdate <= dt and lastdate >= df):
                    corpora.remove(c)
    else:
        # If no date range was provided, use whole date range of the selected corpora
        for c in corpus_data["corpora"]:
            firstdate = corpus_data["corpora"][c]["info"].get("FirstDate")
            lastdate = corpus_data["corpora"][c]["info"].get("LastDate")
            if firstdate and lastdate:
                firstdate = utils.strptime(firstdate.replace("-", "").replace(":", "").replace(" ", ""))
                lastdate = utils.strptime(lastdate.replace("-", "").replace(":", "").replace(" ", ""))

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
        group_by = [(v, True) for v in ("text_datefrom", "text_timefrom", "text_dateto", "text_timeto")]
    else:
        group_by = [(v, True) for v in ("text_datefrom", "text_dateto")]

    if per_corpus:
        # Add zero values for the corpora we removed because of the selected date span
        for corpus in set(corpora_copy).difference(set(corpora)):
            result["corpora"][corpus] = [{"absolute": 0, "relative": 0.0, "sums": {"absolute": 0, "relative": 0.0}}
                                         for _ in range(len(subcqp) + 1)]
            for i, c in enumerate(result["corpora"][corpus][1:]):
                c["cqp"] = subcqp[i]

            if not subcqp:
                result["corpora"][corpus] = result["corpora"][corpus][0]

    # Add zero values for the combined results if no corpora are within the selected date span
    if combined and not corpora:
        result["combined"] = [{"absolute": 0, "relative": 0.0, "sums": {"absolute": 0, "relative": 0.0}}
                              for _ in range(len(subcqp) + 1)]
        for i, c in enumerate(result["combined"][1:]):
            c["cqp"] = subcqp[i]

        if not subcqp:
            result["combined"] = result["combined"][0]

        yield result
        return

    corpora_sizes = {}

    ns = utils.Namespace()
    total_rows = [[] for _ in range(len(subcqp) + 1)]
    ns.total_size = 0

    ns.progress_count = 0
    if incremental:
        yield {"progress_corpora": corpora}

    with ThreadPoolExecutor(max_workers=app.config["PARALLEL_THREADS"]) as executor:
        future_query = dict((executor.submit(count_query_worker, corpus=corpus, cqp=cqp, group_by=group_by,
                                             within=within[corpus],
                                             expand_prequeries=expand_prequeries,
                                             use_cache=args["cache"], cache_max=app.config["CACHE_MAX_STATS"]), corpus)
                            for corpus in corpora)

        for future in futures.as_completed(future_query):
            corpus = future_query[future]
            if future.exception() is not None:
                if "Can't find attribute ``text_datefrom''" not in str(future.exception()):
                    raise utils.CQPError(future.exception())
            else:
                lines, _, corpus_size = future.result()

                corpora_sizes[corpus] = corpus_size
                ns.total_size += corpus_size

                query_no = 0
                for line in lines:
                    if line == utils.END_OF_LINE:
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
                yield {"progress_%d" % ns.progress_count: corpus}
                ns.progress_count += 1

    corpus_timedata = utils.generator_to_dict(
        timespan.timespan({"corpus": corpora, "granularity": granularity, "from": fromdate,
                           "to": todate, "strategy": str(strategy), "cache": args["cache"]},
                          no_combined_cache=True))
    search_timedata = []
    search_timedata_combined = []
    for total_row in total_rows:
        temp = timespan.timespan_calculator(total_row, granularity=granularity, strategy=strategy)
        if per_corpus:
            search_timedata.append(temp["corpora"])
        if combined:
            search_timedata_combined.append(temp["combined"])

    if per_corpus:
        for corpus in corpora:
            corpus_stats = [{"absolute": defaultdict(int),
                             "relative": defaultdict(float),
                             "sums": {"absolute": 0, "relative": 0.0}} for _ in range(len(subcqp) + 1)]

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

    if combined:
        total_stats = [{"absolute": defaultdict(int),
                        "relative": defaultdict(float),
                        "sums": {"absolute": 0, "relative": 0.0}} for _ in range(len(subcqp) + 1)]

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

    yield result


def count_query_worker(corpus, cqp, group_by, within, ignore_case=(), cut=None, expand_prequeries=True,
                       use_cache=False, cache_max=0):
    fullcqp = cqp
    subcqp = None
    if isinstance(cqp[-1], list):
        subcqp = cqp[-1]
        cqp = cqp[:-1]

    if use_cache:
        checksum = utils.get_hash((fullcqp,
                                  group_by,
                                  within,
                                  sorted(ignore_case),
                                  expand_prequeries))

        with memcached.get_client() as mc:
            prefix = utils.cache_prefix(mc, corpus)
            cache_key = "%s:count_data_%s" % (prefix, checksum)
            cache_size_key = "%s:count_size_%s" % (prefix, checksum)

            cached_size = mc.get(cache_size_key)
            if cached_size is not None:
                corpus_hits, corpus_size = cached_size
                if corpus_hits == 0:
                    return [utils.END_OF_LINE] * len(subcqp) if subcqp else [], corpus_hits, corpus_size

                cached_result = mc.get(cache_key)
                if cached_result is not None:
                    return cached_result, corpus_hits, corpus_size

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
            cmd += utils.query_optimize(c, cqpparams_temp, find_match=(not pre_query))[1]
        else:
            cmd += utils.make_query(utils.make_cqp(c, **cqpparams_temp))

        if pre_query:
            cmd += ["Last;"]

    cmd += ["size Last;"]
    cmd += ["info; .EOL.;"]

    # TODO: Match targets in a better way
    has_target = any("@[" in x for x in cqp)

    cmd += ["""tabulate Last %s > "| sort | uniq -c | sort -nr";""" % ", ".join("%s %s%s" % (
        "target" if has_target else ("match" if g[1] else "match .. matchend"), g[0],
        " %c" if g[0] in ignore_case else "") for g in group_by)]

    if subcqp:
        cmd += ["mainresult=Last;"]
        if "expand" in cqpparams_temp:
            del cqpparams_temp["expand"]
        for c in subcqp:
            cmd += [".EOL.;"]
            cmd += ["mainresult;"]
            cmd += utils.query_optimize(c, cqpparams_temp, find_match=True)[1]
            cmd += ["""tabulate Last %s > "| sort | uniq -c | sort -nr";""" % ", ".join(
                "match .. matchend %s" % g[0] for g in group_by)]

    cmd += ["exit;"]

    lines = cwb.run_cqp(cmd)

    # Skip CQP version
    next(lines)

    # Size of the query result
    nr_hits = int(next(lines))

    # Get corpus size
    for line in lines:
        if line.startswith("Size:"):
            _, corpus_size = line.split(":")
            corpus_size = int(corpus_size.strip())
        elif line == utils.END_OF_LINE:
            break

    if use_cache:
        lines = list(lines)
        with memcached.get_client() as mc:
            mc.add(cache_size_key, (nr_hits, corpus_size))

            # Only save actual data if number of lines doesn't exceed the limit
            if len(lines) <= cache_max:
                lines = tuple(lines)
                try:
                    mc.add(cache_key, lines)
                except MemcacheError:
                    pass

    return lines, nr_hits, corpus_size


def count_query_worker_simple(corpus, cqp, group_by, within=None, ignore_case=(), expand_prequeries=True,
                              use_cache=False, cache_max=None):
    """Worker for simple statistics queries which can be run using cwb-scan-corpus.
    Currently only used for searches on [] (any word)."""
    lines = list(cwb.run_cwb_scan(corpus, [g[0] for g in group_by]))
    nr_hits = 0

    ic_index = []
    new_lines = {}
    if ignore_case:
        ic_index = [i for i, g in enumerate(group_by) if g[0] in ignore_case]

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

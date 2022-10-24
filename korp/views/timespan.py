import itertools
from collections import defaultdict

from dateutil.relativedelta import relativedelta
from flask import Blueprint
from flask import current_app as app
from pymemcache.exceptions import MemcacheError

from korp import utils
from korp.db import mysql
from korp.memcached import memcached

bp = Blueprint("timespan", __name__)


@bp.route("/timespan", methods=["GET", "POST"])
@utils.main_handler
@utils.prevent_timeout
def timespan(args, no_combined_cache=False):
    """Calculate timespan information for corpora.
    The time information is retrieved from the database.
    """
    utils.assert_key("corpus", args, utils.IS_IDENT, True)
    utils.assert_key("granularity", args, r"[ymdhnsYMDHNS]")
    utils.assert_key("combined", args, r"(true|false)")
    utils.assert_key("per_corpus", args, r"(true|false)")
    utils.assert_key("strategy", args, r"^[123]$")
    utils.assert_key("from", args, r"^(\d{8}\d{6}?|\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?)$")
    utils.assert_key("to", args, r"^(\d{8}\d{6}?|\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?)$")

    corpora = utils.parse_corpora(args)
    # check_authorization(corpora)

    granularity = (args.get("granularity") or "y").lower()
    combined = utils.parse_bool(args, "combined", True)
    per_corpus = utils.parse_bool(args, "per_corpus", True)
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
        combined_checksum = utils.get_hash((granularity,
                                           combined,
                                           per_corpus,
                                           fromdate,
                                           todate,
                                           sorted(corpora)))
        with memcached.get_client() as mc:
            cache_combined_key = "%s:timespan_%s" % (utils.cache_prefix(mc), utils.get_hash(combined_checksum))
            result = mc.get(cache_combined_key)
        if result is not None:
            if "debug" in args:
                result.setdefault("DEBUG", {})
                result["DEBUG"]["cache_read"] = True
            yield result
            return

        # Look for per-corpus caches
        with memcached.get_client() as mc:
            cache_prefixes = utils.cache_prefix(mc, corpora)
            for corpus in corpora:
                corpus_checksum = utils.get_hash((fromdate, todate, granularity, strategy))
                cache_key = "%s:timespan_%s" % (cache_prefixes[corpus], corpus_checksum)
                corpus_cached_data = mc.get(cache_key)

                if corpus_cached_data is not None:
                    cached_data.extend(corpus_cached_data)
                    corpora_rest.remove(corpus)

    ns = {}

    with app.app_context():
        if corpora_rest:
            corpora_sql = "(%s)" % ", ".join("'%s'" % utils.sql_escape(c) for c in corpora_rest)
            fromto = ""

            if strategy == 1:
                if fromdate and todate:
                    fromto = " AND ((datefrom >= %s AND dateto <= %s) OR (datefrom <= %s AND dateto >= %s))" % (
                        utils.sql_escape(fromdate), utils.sql_escape(todate), utils.sql_escape(fromdate), utils.sql_escape(todate))
            elif strategy == 2:
                if todate:
                    fromto += " AND datefrom <= '%s'" % utils.sql_escape(todate)
                if fromdate:
                    fromto = " AND dateto >= '%s'" % utils.sql_escape(fromdate)
            elif strategy == 3:
                if fromdate:
                    fromto = " AND datefrom >= '%s'" % utils.sql_escape(fromdate)
                if todate:
                    fromto += " AND dateto <= '%s'" % utils.sql_escape(todate)

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
            def save_cache(mc, corpus, data):
                corpus_checksum = utils.get_hash((fromdate, todate, granularity, strategy))
                cache_key = "%s:timespan_%s" % (cache_prefixes[corpus], corpus_checksum)
                try:
                    mc.add(cache_key, data)
                except MemcacheError:
                    pass

            corpus = None
            corpus_data = []
            with memcached.get_client() as mc:
                for row in cursor:
                    if corpus is None:
                        corpus = row["corpus"]
                    elif not row["corpus"] == corpus:
                        save_cache(mc, corpus, corpus_data)
                        corpus_data = []
                        corpus = row["corpus"]
                    corpus_data.append(row)
                    cached_data.append(row)
                if corpus is not None:
                    save_cache(mc, corpus, corpus_data)

        ns["result"] = timespan_calculator(itertools.chain(cached_data, cursor), granularity=granularity,
                                           combined=combined, per_corpus=per_corpus, strategy=strategy)

        if corpora_rest:
            cursor.close()

    if args["cache"] and not no_combined_cache:
        # Save cache for whole query
        try:
            with memcached.get_client() as mc:
                mc.add(cache_combined_key, ns["result"])
        except MemcacheError:
            pass

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

    def plusminusone(date, value, df, negative=False):
        date = "0" + date if len(date) % 2 else date  # Handle years with three digits
        d = utils.strptime(date)
        if negative:
            d = d - value
        else:
            d = d + value
        return int(d.strftime(df))

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

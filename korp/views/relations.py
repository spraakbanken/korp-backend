import itertools
import math
import time
from collections import defaultdict
from copy import deepcopy

from flask import Blueprint
from flask import current_app as app
from pymemcache.exceptions import MemcacheError

from korp import utils
from korp.db import mysql
from korp.memcached import memcached
from . import query

bp = Blueprint("relations", __name__)


@bp.route("/relations", methods=["GET", "POST"])
@utils.main_handler
@utils.prevent_timeout
def relations(args):
    """Calculate word picture data."""
    utils.assert_key("corpus", args, utils.IS_IDENT, True)
    utils.assert_key("word", args, "", True)
    utils.assert_key("type", args, r"(word|lemgram)", False)
    utils.assert_key("min", args, utils.IS_NUMBER, False)
    utils.assert_key("max", args, utils.IS_NUMBER, False)
    utils.assert_key("incremental", args, r"(true|false)")

    corpora = utils.parse_corpora(args)
    utils.check_authorization(corpora)

    incremental = utils.parse_bool(args, "incremental", False)

    word = args.get("word")
    search_type = args.get("type", "")
    minfreq = args.get("min")
    sort = args.get("sort") or "mi"
    maxresults = int(args.get("max") or 15)
    minfreqsql = " AND freq >= %s" % minfreq if minfreq else ""

    result = {}

    with app.app_context():
        cursor = mysql.connection.cursor()
        cursor.execute("SET @@session.long_query_time = 1000;")

        # Get available tables
        cursor.execute("SHOW TABLES LIKE '" + app.config["DBWPTABLE"] + "_%';")
        tables = set(list(x.values())[0] for x in cursor)
        # Filter out corpora which don't exist in database
        corpora = [x for x in corpora if app.config["DBWPTABLE"] + "_" + x.upper() in tables]
        if not corpora:
            yield {}
            return

        relations_data = []
        corpora_rest = corpora[:]

        if args["cache"]:
            with memcached.get_client() as mc:
                cache_prefixes = utils.cache_prefix(mc, corpora)
                memcached_keys = {}
                for corpus in corpora:
                    corpus_checksum = utils.get_hash((word,
                                                     search_type,
                                                     minfreq))
                    memcached_keys["%s:relations_%s" % (cache_prefixes[corpus], corpus_checksum)] = corpus

                cached_data = mc.get_many(memcached_keys.keys())

            for key in cached_data:
                relations_data.extend(cached_data[key])
                corpora_rest.remove(memcached_keys[key])

        selects = []

        if search_type == "lemgram":
            lemgram_sql = "'%s'" % utils.sql_escape(word)

            for corpus in corpora_rest:
                corpus_sql = "'%s'" % utils.sql_escape(corpus).upper()
                corpus_table = app.config["DBWPTABLE"] + "_" + corpus.upper()

                selects.append(
                    (
                        corpus.upper(),
                        "(SELECT S1.string AS head, S1.pos AS headpos, F.rel, S2.string AS dep, S2.pos AS deppos, "
                        "S2.stringextra AS depextra, F.freq, R.freq AS rel_freq, HR.freq AS head_rel_freq, "
                        f"DR.freq AS dep_rel_freq, {corpus_sql} AS corpus, F.id " +
                        f"FROM `{corpus_table}_strings` AS S1, `{corpus_table}_strings` AS S2, `{corpus_table}` AS F, "
                        f"`{corpus_table}_rel` AS R, `{corpus_table}_head_rel` AS HR, `{corpus_table}_dep_rel` AS DR "
                        f"WHERE S1.string = {lemgram_sql} AND F.head = S1.id AND S2.id = F.dep {minfreqsql} "
                        "AND F.bfhead = 1 AND F.bfdep = 1 AND F.rel = R.rel AND F.head = HR.head AND F.rel = HR.rel "
                        "AND F.dep = DR.dep AND F.rel = DR.rel)"
                    )
                )
                selects.append(
                    (
                        None,
                        "(SELECT S1.string AS head, S1.pos AS headpos, F.rel, S2.string AS dep, S2.pos AS deppos, "
                        "S2.stringextra AS depextra, F.freq, R.freq AS rel_freq, HR.freq AS head_rel_freq, "
                        f"DR.freq AS dep_rel_freq, {corpus_sql} AS corpus, F.id "
                        f"FROM `{corpus_table}_strings` AS S1, `{corpus_table}_strings` AS S2, `{corpus_table}` AS F, "
                        f"`{corpus_table}_rel` AS R, `{corpus_table}_head_rel` AS HR, `{corpus_table}_dep_rel` AS DR "
                        f"WHERE S2.string = {lemgram_sql} AND F.dep = S2.id AND S1.id = F.head {minfreqsql} "
                        "AND F.bfhead = 1 AND F.bfdep = 1 AND F.rel = R.rel AND F.head = HR.head AND F.rel = HR.rel "
                        "AND F.dep = DR.dep AND F.rel = DR.rel)"
                    )
                )
        else:
            word_sql = "'%s'" % utils.sql_escape(word)
            word = word

            for corpus in corpora_rest:
                corpus_sql = "'%s'" % utils.sql_escape(corpus).upper()
                corpus_table = app.config["DBWPTABLE"] + "_" + corpus.upper()

                selects.append(
                    (
                        corpus.upper(),
                        "(SELECT S1.string AS head, S1.pos AS headpos, F.rel, S2.string AS dep, S2.pos AS deppos, "
                        "S2.stringextra AS depextra, F.freq, R.freq AS rel_freq, HR.freq AS head_rel_freq, "
                        f"DR.freq AS dep_rel_freq, {corpus_sql} AS corpus, F.id "
                        f"FROM `{corpus_table}_strings` AS S1, `{corpus_table}_strings` AS S2, `{corpus_table}` AS F, "
                        f"`{corpus_table}_rel` AS R, `{corpus_table}_head_rel` AS HR, `{corpus_table}_dep_rel` AS DR "
                        f"WHERE S1.string = {word_sql} AND F.head = S1.id AND F.wfhead = 1 AND S2.id = F.dep "
                        f"{minfreqsql} AND F.rel = R.rel AND F.head = HR.head AND F.rel = HR.rel AND F.dep = DR.dep "
                        "AND F.rel = DR.rel)"
                    )
                )
                selects.append(
                    (
                        None,
                        "(SELECT S1.string AS head, S1.pos AS headpos, F.rel, S2.string AS dep, S2.pos AS deppos, "
                        "S2.stringextra AS depextra, F.freq, R.freq AS rel_freq, HR.freq AS head_rel_freq, "
                        f"DR.freq AS dep_rel_freq, {corpus_sql} AS corpus, F.id "
                        f"FROM `{corpus_table}_strings` AS S1, `{corpus_table}_strings` AS S2, `{corpus_table}` AS F, "
                        f"`{corpus_table}_rel` AS R, `{corpus_table}_head_rel` AS HR, `{corpus_table}_dep_rel` AS DR "
                        f"WHERE S2.string = {word_sql} AND F.dep = S2.id AND F.wfdep = 1 AND S1.id = F.head "
                        f"{minfreqsql} AND F.rel = R.rel AND F.head = HR.head AND F.rel = HR.rel AND F.dep = DR.dep "
                        "AND F.rel = DR.rel)"
                    )
                )

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

    def save_cache(corpus, data):
        corpus_checksum = utils.get_hash((word, search_type, minfreq))
        with memcached.get_client() as mc:
            try:
                mc.add("%s:relations_%s" % (cache_prefixes[corpus], corpus_checksum), data)
            except MemcacheError:
                pass

    for row in itertools.chain(relations_data, (None,), cursor_result):
        if row is None:
            if args["cache"]:
                # Start caching results
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

    sortedrels = sorted(rels.items(), key=lambda x: (x[0][1], x[1][sort]), reverse=True)

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


@bp.route("/relations_sentences", methods=["GET", "POST"])
@utils.main_handler
@utils.prevent_timeout
def relations_sentences(args):
    """Execute a CQP query to find sentences with a given relation from a word picture."""
    utils.assert_key("source", args, "", True)
    utils.assert_key("start", args, utils.IS_NUMBER, False)
    utils.assert_key("end", args, utils.IS_NUMBER, False)

    temp_source = args.get("source")
    if isinstance(temp_source, str):
        temp_source = temp_source.split(utils.QUERY_DELIM)
    source = defaultdict(set)
    for s in temp_source:
        c, i = s.split(":")
        source[c].add(i)

    utils.check_authorization(source.keys())

    start = int(args.get("start") or 0)
    end = int(args.get("end") or 9)
    shown = args.get("show") or "word"
    shown_structs = args.get("show_struct") or []
    if isinstance(shown_structs, str):
        shown_structs = shown_structs.split(utils.QUERY_DELIM)
    shown_structs = set(shown_structs)

    default_context = args.get("default_context") or "1 sentence"

    querystarttime = time.time()

    with app.app_context():
        cursor = mysql.connection.cursor()
        cursor.execute("SET @@session.long_query_time = 1000;")
        selects = []
        counts = []

        # Get available tables
        cursor.execute("SHOW TABLES LIKE '" + app.config["DBWPTABLE"] + "_%';")
        tables = set(list(x.values())[0] for x in cursor)
        # Filter out corpora which doesn't exist in database
        source = sorted([x for x in iter(source.items()) if app.config["DBWPTABLE"] + "_" + x[0].upper() in tables])
        if not source:
            yield {}
            return
        corpora = [x[0] for x in source]

        for s in source:
            corpus, ids = s
            ids = [int(i) for i in ids]
            ids_list = "(" + ", ".join("%d" % i for i in ids) + ")"

            corpus_table_sentences = app.config["DBWPTABLE"] + f"_{corpus.upper()}_sentences"

            selects.append(
                f"(SELECT S.sentence, S.start, S.end, '{utils.sql_escape(corpus.upper())}' AS corpus "
                f"FROM `{corpus_table_sentences}` as S "
                f"WHERE S.id IN {ids_list})"
            )
            counts.append(
                f"(SELECT '{utils.sql_escape(corpus.upper())}' AS corpus, COUNT(*) AS freq "
                f"FROM `{corpus_table_sentences}` as S "
                f"WHERE S.id IN {ids_list})"
            )

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
            corpora_dict.setdefault(row["corpus"], {}).setdefault(row["sentence"], []).append(
                (row["start"], row["end"]))

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
             "default_context": default_context}
        if shown:
            q["show"] = shown
        result_temp = utils.generator_to_dict(query.query(q))

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

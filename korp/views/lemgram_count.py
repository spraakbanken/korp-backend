from flask import Blueprint
from flask import current_app as app

from korp import utils
from korp.db import mysql

bp = Blueprint("lemgram_count", __name__)


@bp.route("/lemgram_count", methods=["GET", "POST"])
@utils.main_handler
@utils.prevent_timeout
def lemgram_count(args):
    """Return lemgram statistics per corpus."""
    utils.assert_key("lemgram", args, r"", True)
    utils.assert_key("corpus", args, utils.IS_IDENT)
    utils.assert_key("count", args, r"(lemgram|prefix|suffix)")

    corpora = utils.parse_corpora(args)
    utils.check_authorization(corpora)

    lemgram = args.get("lemgram")
    if isinstance(lemgram, str):
        lemgram = lemgram.split(utils.QUERY_DELIM)
    lemgram = set(lemgram)

    count = args.get("count") or "lemgram"
    if isinstance(count, str):
        count = count.split(utils.QUERY_DELIM)
    count = set(count)

    counts = {"lemgram": "freq",
              "prefix": "freq_prefix",
              "suffix": "freq_suffix"}

    sums = " + ".join("SUM(%s)" % counts[c] for c in count)

    lemgram_sql = " lemgram IN (%s)" % ", ".join("'%s'" % utils.sql_escape(l) for l in lemgram)
    corpora_sql = " AND corpus IN (%s)" % ", ".join("'%s'" % utils.sql_escape(c) for c in corpora) if corpora else ""

    sql = "SELECT lemgram, " + sums + " AS freq FROM lemgram_index WHERE" + lemgram_sql + corpora_sql + \
          " GROUP BY lemgram;"

    result = {}
    cursor = mysql.connection.cursor()
    cursor.execute(sql)

    for row in cursor:
        # We need this check here, since a search for "hår" also returns "här" and "har".
        if row["lemgram"] in lemgram and int(row["freq"]) > 0:
            result[row["lemgram"]] = int(row["freq"])

    cursor.close()

    yield result

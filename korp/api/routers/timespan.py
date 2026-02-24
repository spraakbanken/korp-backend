"""Router for timespan information."""

import itertools
from collections import defaultdict
from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Annotated, Any

from dateutil.relativedelta import relativedelta
from fastapi import APIRouter, Query
from sqlalchemy import text

from korp import utils
from korp.api import params
from korp.api.params import GranularityValues
from korp.memcached import CacheError

router = APIRouter(tags=["Statistics"])


@router.get("/timespan", response_model=None)
@router.post("/timespan", response_model=None, include_in_schema=False)
@utils.api_handler
async def timespan(
    ctx: utils.CtxDep,
    corpus: params.CorpusParam,
    granularity: params.GranularityParam = GranularityValues.year,
    combined: params.CombinedParam = True,
    per_corpus: params.PerCorpusParam = True,
    strategy: params.StrategyParam = params.StrategyValues.some_overlaps,
    date_from: Annotated[
        str | None,
        Query(
            alias="from",
            pattern=r"^(\d{8}\d{6}?|\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?)$",
            description="Start date for filtering (inclusive). Format: YYYYMMDD[HHMMSS] or YYYY-MM-DD[ HH:MM:SS].",
        ),
    ] = None,
    date_to: Annotated[
        str | None,
        Query(
            alias="to",
            pattern=r"^(\d{8}\d{6}?|\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?)$",
            description="End date for filtering (inclusive). Format: YYYYMMDD[HHMMSS] or YYYY-MM-DD[ HH:MM:SS].",
        ),
    ] = None,
) -> AsyncIterator[dict]:
    """Calculate timespan information for corpora.

    Args:
        ctx: The request context.
        corpus: Comma-separated list of corpora.
        granularity: Granularity of result ('y' = year, 'm' = month, 'd' = day, 'h' = hour, 'n' = minute, 's' = second).
        combined: Whether to include combined results.
        per_corpus: Whether to include results per corpus.
        strategy: Strategy for date range matching (1 = some overlaps permitted, 2 = all overlaps permitted, 3 = strict
            matching).
        date_from: Start date for filtering (inclusive).
        date_to: End date for filtering (inclusive).

    Yields:
        A dictionary containing the timespan information.
    """
    corpora = corpus or []
    # check_authorization(corpora)

    yield await get_timespan(
        ctx,
        corpora,
        granularity=granularity,
        combined=combined,
        per_corpus=per_corpus,
        strategy=strategy,
        date_from=date_from,
        date_to=date_to,
    )


async def get_timespan(
    ctx: utils.CtxDep,
    corpora: list[str],
    granularity: GranularityValues = GranularityValues.year,
    combined: bool = True,
    per_corpus: bool = True,
    strategy: params.StrategyValues = params.StrategyValues.some_overlaps,
    date_from: str | None = None,
    date_to: str | None = None,
    no_combined_cache: bool = False,
) -> dict:
    """Calculate timespan information for corpora.

    Args:
        ctx: The request context.
        corpora: List of corpora.
        granularity: Granularity of result ('y' = year, 'm' = month, 'd' = day, 'h' = hour, 'n' = minute, 's' = second).
        combined: Whether to include combined results.
        per_corpus: Whether to include results per corpus.
        strategy: Strategy for date range matching (1 = some overlaps permitted, 2 = all overlaps permitted, 3 = strict
            matching).
        date_from: Start date for filtering (inclusive).
        date_to: End date for filtering (inclusive).
        no_combined_cache: If True, do not use combined caching for multiple corpora.

    Returns:
        A dictionary containing the timespan information.

    Raises:
        ValueError: If only one of date_from or date_to is provided.
    """
    if (date_from or date_to) and not (date_from and date_to):
        raise ValueError("When using 'from' or 'to', both need to be specified.")

    # Mapping from GranularityValues to number of characters to shorten date strings in the database to
    shorten = {
        GranularityValues.year: 4,
        GranularityValues.month: 7,
        GranularityValues.day: 10,
        GranularityValues.hour: 13,
        GranularityValues.minute: 16,
        GranularityValues.second: 19,
    }

    cached_data = []
    corpora_rest = corpora.copy()

    if ctx.common.cache:
        # Check if whole query is cached
        combined_checksum = utils.get_hash(
            (granularity, strategy, combined, per_corpus, date_from, date_to, sorted(corpora))
        )
        cache_prefix = await utils.cache_prefix(ctx.cache)
        cache_combined_key = f"{cache_prefix}:timespan_{combined_checksum}"
        result = await ctx.cache.get(cache_combined_key)
        if result is not None:
            if ctx.common.debug:
                result.setdefault("DEBUG", {})
                result["DEBUG"]["cache_read"] = True
            return result

        # Look for per-corpus caches
        cache_prefixes = await utils.cache_prefix(ctx.cache, corpora)
        for c in corpora:
            corpus_checksum = utils.get_hash((date_from, date_to, granularity, strategy))
            cache_key = f"{cache_prefixes[c]}:timespan_{corpus_checksum}"
            corpus_cached_data = await ctx.cache.get(cache_key)
            if corpus_cached_data is not None:
                cached_data.extend(corpus_cached_data)
                corpora_rest.remove(c)

    if corpora_rest:
        corpora_sql = "({})".format(", ".join(f"'{utils.sql_escape(c)}'" for c in corpora_rest))
        fromto = ""

        if strategy == params.StrategyValues.some_overlaps:
            if date_from and date_to:
                fromto = (
                    f" AND ((datefrom >= '{utils.sql_escape(date_from)}'"
                    f" AND dateto <= '{utils.sql_escape(date_to)}')"
                    f" OR (datefrom <= '{utils.sql_escape(date_from)}'"
                    f" AND dateto >= '{utils.sql_escape(date_to)}'))"
                )
        elif strategy == params.StrategyValues.all_overlaps:
            if date_to:
                fromto = f" AND datefrom <= '{utils.sql_escape(date_to)}'"
            if date_from:
                fromto += f" AND dateto >= '{utils.sql_escape(date_from)}'"
        elif strategy == params.StrategyValues.strict:
            if date_from:
                fromto = f" AND datefrom >= '{utils.sql_escape(date_from)}'"
            if date_to:
                fromto += f" AND dateto <= '{utils.sql_escape(date_to)}'"

        # TODO: Skip grouping on corpus when we're only after the combined results.
        # We do the granularity truncation and summation in the DB query if we can (depending on strategy),
        # since it's much faster than doing it afterwards

        timedata_corpus = (
            "timedata_date"
            if granularity in {GranularityValues.year, GranularityValues.month, GranularityValues.day}
            else "timedata"
        )
        if strategy == params.StrategyValues.some_overlaps:
            # We need the full dates for this strategy, so no truncating of the results
            # We cast datefrom/dateto to CHAR to avoid issues with year zero
            sql = (
                "SELECT corpus, CAST(datefrom AS CHAR) AS df, CAST(dateto AS CHAR) AS dt, SUM(tokens) AS sum FROM "
                + timedata_corpus
                + " WHERE corpus IN "
                + corpora_sql
                + fromto
                + " GROUP BY corpus, df, dt ORDER BY NULL;"  # Avoid implicit ordering in older MySQL versions
            )
        else:
            sql = (
                "SELECT corpus, LEFT(datefrom, "
                + str(shorten[granularity])
                + ") AS df, LEFT(dateto, "
                + str(shorten[granularity])
                + ") AS dt, SUM(tokens) AS sum FROM "
                + timedata_corpus
                + " WHERE corpus IN "
                + corpora_sql
                + fromto
                + " GROUP BY corpus, df, dt ORDER BY NULL;"
            )

        async with ctx.db.async_connection() as conn:
            try:
                result = await conn.execute(text(sql))
                rows_result = result.mappings().all()
                rows = [dict(row) for row in rows_result] if ctx.common.cache else rows_result
            except Exception:
                await conn.invalidate()
                raise
    else:
        rows = []

    if ctx.common.cache:

        async def save_cache(corpus: str, data: list[Mapping[str, Any]]) -> None:
            corpus_checksum = utils.get_hash((date_from, date_to, granularity, strategy))
            cache_key = f"{cache_prefixes[corpus]}:timespan_{corpus_checksum}"
            try:
                await ctx.cache.add(cache_key, data)
            except CacheError:
                pass

        corpus_data: defaultdict[str, list[Mapping[Any, Any]]] = defaultdict(list)
        for row in rows:
            corpus_data[row["corpus"]].append(row)
        for corpus, data in corpus_data.items():
            await save_cache(corpus, data)

    result = timespan_calculator(
        itertools.chain(cached_data, rows),
        granularity=granularity,
        combined=combined,
        per_corpus=per_corpus,
        strategy=strategy,
    )

    if ctx.common.cache and not no_combined_cache:
        # Save cache for whole query
        try:
            await ctx.cache.add(cache_combined_key, result)
        except CacheError:
            pass

    return result


def timespan_calculator(
    timedata: Iterable[Mapping],
    granularity: GranularityValues = GranularityValues.year,
    combined: bool = True,
    per_corpus: bool = True,
    strategy: params.StrategyValues = params.StrategyValues.some_overlaps,
) -> dict:
    """Calculate timespan information for corpora.

    Args:
        timedata: List of time data dictionaries with keys 'corpus', 'df' (datefrom), 'dt' (dateto), and 'sum' (token
            count).
        granularity: Granularity of result ('y' = year, 'm' = month, 'd' = day, 'h' = hour, 'n' = minute, 's' = second).
        combined: Whether to include combined results.
        per_corpus: Whether to include results per corpus.
        strategy: Strategy for date range matching (1 = some overlaps permitted, 2 = all overlaps permitted, 3 = strict
            matching).

    Returns:
        A dictionary containing the timespan information.
    """
    gs = {
        GranularityValues.year: 4,
        GranularityValues.month: 6,
        GranularityValues.day: 8,
        GranularityValues.hour: 10,
        GranularityValues.minute: 12,
        GranularityValues.second: 14,
    }

    def plusminusone(date: str, value: relativedelta, df: str, negative: bool = False) -> int:
        """Add or subtract one unit of the given granularity to/from the date.

        Args:
            date: The date string.
            value: The relativedelta value to add or subtract.
            df: The date format string.
            negative: If True, subtract the value; otherwise, add it.

        Returns:
            The modified date as an integer.
        """
        date = "0" + date if len(date) % 2 else date  # Handle years with three digits
        d = utils.strptime(date)
        if negative:
            d -= value
        else:
            d += value
        return int(d.strftime(df))

    def shorten(date: str, g: GranularityValues) -> int:
        """Return a shortened version of the date according to the granularity."""
        alt = 1 if len(date) % 2 else 0  # Handle years with three digits
        return int(date[: gs[g] - alt])

    if granularity == GranularityValues.year:
        df = "%Y"
        add = relativedelta(years=1)
    elif granularity == GranularityValues.month:
        df = "%Y%m"
        add = relativedelta(months=1)
    elif granularity == GranularityValues.day:
        df = "%Y%m%d"
        add = relativedelta(days=1)
    elif granularity == GranularityValues.hour:
        df = "%Y%m%d%H"
        add = relativedelta(hours=1)
    elif granularity == GranularityValues.minute:
        df = "%Y%m%d%H%M"
        add = relativedelta(minutes=1)
    elif granularity == GranularityValues.second:
        df = "%Y%m%d%H%M%S"
        add = relativedelta(seconds=1)

    rows = defaultdict(list)
    nodes: defaultdict[str, set[tuple[str, int]]] = defaultdict(set)

    datemin = (
        "00000101"
        if granularity in {GranularityValues.year, GranularityValues.month, GranularityValues.day}
        else "00000101000000"
    )
    datemax = (
        "99991231"
        if granularity in {GranularityValues.year, GranularityValues.month, GranularityValues.day}
        else "99991231235959"
    )

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

        if strategy == params.StrategyValues.some_overlaps:
            # Some overlaps permitted
            # (t1 >= t1' AND t2 <= t2') OR (t1 <= t1' AND t2 >= t2')
            if datefrom_short != dateto_short:
                if datefrom[gs[granularity] :] != datemin[gs[granularity] :]:
                    # Add 1 to datefrom_short
                    datefrom_short = plusminusone(str(datefrom_short), add, df)

                if dateto[gs[granularity] :] != datemax[gs[granularity] :]:
                    # Subtract 1 from dateto_short
                    dateto_short = plusminusone(str(dateto_short), add, df, negative=True)

                # Check that datefrom is still before dateto
                if not datefrom < dateto:
                    continue
        elif strategy == params.StrategyValues.all_overlaps:
            # All overlaps permitted
            # t1 <= t2' AND t2 >= t1'
            pass
        elif strategy == params.StrategyValues.strict:  # noqa: SIM102
            # Strict matching. No overlaps tolerated.
            # t1 >= t1' AND t2 <= t2'

            if datefrom_short != dateto_short:
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

    corpusnodes = {k: sorted(v, key=lambda x: (x[1] or 0, x[0])) for k, v in nodes.items()}
    result = {}
    if per_corpus:
        result["corpora"] = {}
    if combined:
        result["combined"] = {}

    for corpus, nodes_ in corpusnodes.items():
        data = defaultdict(int)

        for i in range(len(nodes_) - 1):
            start = nodes_[i]
            end = nodes_[i + 1]
            if start[0] == "t":
                start = plusminusone(str(start[1]), add, df) if start[1] else 0
                if start == end[1] and end[0] == "f":
                    continue
            else:
                start = start[1]

            end = 0 if not end[1] else end[1] if end[0] == "t" else plusminusone(str(end[1]), add, df, True)

            if start:
                data[str(start)] = 0

            for row in rows[corpus]:
                if row["datefrom"] <= start and row["dateto"] >= end:
                    data[str(start or "")] += row["freq"]

            if end:
                data[str(plusminusone(str(end), add, df, False))] = 0

        if combined and corpus == "__combined__":
            result["combined"] = data
        else:
            result["corpora"][corpus] = data

    return result

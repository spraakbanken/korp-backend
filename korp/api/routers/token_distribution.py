"""Router for token distribution information."""

import bisect
import functools
import itertools
from collections import defaultdict
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass
from logging import getLogger
from operator import itemgetter
from time import perf_counter
from typing import Annotated, Any, TypeAlias

import anyio.to_process
import anyio.to_thread
from dateutil.relativedelta import relativedelta
from fastapi import APIRouter, Query
from pydantic import ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema
from sqlalchemy import text

from korp import caching, utils
from korp.api import params, schemas
from korp.api.params import GranularityValues
from korp.config import settings
from korp.dependencies import CtxDep
from korp.handler import api_handler, docs_response
from korp.memcached import CacheError

router = APIRouter(tags=["Statistics"])
logger = getLogger(__name__)

TOKEN_DISTRIBUTION_DESCRIPTION = f"""Show the distribution of corpus tokens over time.

The route returns token counts grouped by time period. Use `granularity` to choose the period size: year, month, day,
hour, minute, or second. The response can include per-corpus series, one combined series for all selected corpora, or
both.

Each key in a time series marks the start of a period. The value applies from that key until the next key. For example,
with yearly granularity, a series containing `2010: 100`, `2012: 50`, and `2015: 0` means 100 tokens during 2010-2011,
50 tokens during 2012-2014, and zero tokens from 2015 until the next key.

Use `date_from` and `date_to` together to limit the date range.

### Time Matching Strategies

{params.TIME_STRATEGY_DESCRIPTION}

### Example

Show yearly token distribution for a corpus:

`/token_distribution?corpus=VIVILL&granularity=year`
"""

DateFromParam: TypeAlias = Annotated[
    str | None,
    Query(
        pattern=r"^(\d{8}(\d{6})?|\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?)$",
        description=(
            "Start date/time for filtering, inclusive. Must be used together with `date_to`. Accepted formats: "
            "YYYYMMDDHHMMSS, YYYYMMDD, YYYY-MM-DD HH:MM:SS, or YYYY-MM-DD."
        ),
        examples=["20200101000000", "2020-01-01"],
    ),
]

DateToParam: TypeAlias = Annotated[
    str | None,
    Query(
        pattern=r"^(\d{8}(\d{6})?|\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?)$",
        description=(
            "End date/time for filtering, inclusive. Must be used together with `date_from`. Accepted formats: "
            "YYYYMMDDHHMMSS, YYYYMMDD, YYYY-MM-DD HH:MM:SS, or YYYY-MM-DD."
        ),
        examples=["20201231235959", "2020-12-31"],
    ),
]


class TokenDistributionResponse(schemas.CommonResponse):
    """Response model for `/token_distribution` route."""

    model_config = ConfigDict(extra="allow")

    corpora: dict[str, dict[str, int]] | SkipJsonSchema[None] = Field(
        None,
        description=(
            "Token counts per time period, keyed first by corpus id and then by period start. Omitted when "
            "`per_corpus=false`."
        ),
        examples=[{"ROMI": {"2017": 15366, "2018": 7437}}],
    )
    combined: dict[str, int] | SkipJsonSchema[None] = Field(
        None,
        description="Combined token counts per time period across all selected corpora. Omitted when `combined=false`.",
        examples=[{"2017": 15366, "2018": 7437}],
    )


@dataclass(frozen=True, slots=True)
class _GranularityConfig:
    """Precomputed configuration for a single granularity level."""

    sql_left_len: int
    """Number of characters to keep with SQL LEFT() (includes separators, e.g. 'YYYY-MM' = 7)."""
    digit_len: int
    """Number of digits in the shortened date string (e.g. 'YYYYMM' = 6)."""
    date_fmt: str
    """strftime format for the granularity level."""
    delta: relativedelta
    """One unit of this granularity as a relativedelta."""
    uses_date_table: bool
    """Whether to use the 'timedata_date' table (True) or 'timedata' (False)."""


_GRANULARITY: dict[GranularityValues, _GranularityConfig] = {
    GranularityValues.year: _GranularityConfig(4, 4, "%Y", relativedelta(years=1), True),
    GranularityValues.month: _GranularityConfig(7, 6, "%Y%m", relativedelta(months=1), True),
    GranularityValues.day: _GranularityConfig(10, 8, "%Y%m%d", relativedelta(days=1), True),
    GranularityValues.hour: _GranularityConfig(13, 10, "%Y%m%d%H", relativedelta(hours=1), False),
    GranularityValues.minute: _GranularityConfig(16, 12, "%Y%m%d%H%M", relativedelta(minutes=1), False),
    GranularityValues.second: _GranularityConfig(19, 14, "%Y%m%d%H%M%S", relativedelta(seconds=1), False),
}


def _digits_only(value: Any) -> str:
    """Extract only digit characters from a value.

    Returns an empty string if the value is falsy or contains only zeros.

    Returns:
        A string containing only the digit characters, or an empty string.
    """
    if not value:
        return ""
    result = "".join(c for c in str(value) if c.isdigit())
    if not result or not result.strip("0"):
        return ""
    return result


@functools.lru_cache(maxsize=4096)
def _adjust_date(date_str: str, granularity: GranularityValues, *, subtract: bool = False) -> int:
    """Adjust a date by adding or subtracting one granularity unit.

    Args:
        date_str: The date string (digits only).
        granularity: The granularity level determining the delta and format.
        subtract: If True, subtract the delta; otherwise add it.

    Returns:
        The adjusted date as an integer.
    """
    g_config = _GRANULARITY[granularity]
    padded = "0" + date_str if len(date_str) % 2 else date_str
    d = utils.strptime(padded)
    if subtract:
        d -= g_config.delta
    else:
        d += g_config.delta
    return int(d.strftime(g_config.date_fmt))


@router.get(
    "/token_distribution",
    response_model=None,
    responses=docs_response(TokenDistributionResponse),
    summary="Token Distribution",
    description=TOKEN_DISTRIBUTION_DESCRIPTION,
)
@router.post("/token_distribution", response_model=None, include_in_schema=False)
@api_handler
async def token_distribution(
    ctx: CtxDep,
    corpus: params.CorpusParam,
    granularity: params.GranularityParam = GranularityValues.year,
    combined: params.CombinedParam = True,
    per_corpus: params.PerCorpusParam = True,
    strategy: params.StrategyParam = params.StrategyValues.some_overlaps,
    date_from: DateFromParam = None,
    date_to: DateToParam = None,
) -> AsyncIterator[dict]:
    """Calculate token distribution information for corpora.

    Args:
        ctx: The request context.
        corpus: Comma-separated list of corpora.
        granularity: Granularity of result.
        combined: Whether to include combined results.
        per_corpus: Whether to include results per corpus.
        strategy: Strategy for date range matching.
        date_from: Start date for filtering (inclusive).
        date_to: End date for filtering (inclusive).

    Yields:
        A dictionary containing the token distribution information.
    """
    corpora = corpus or []

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
    ctx: CtxDep,
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
        granularity: Granularity of result.
        combined: Whether to include combined results.
        per_corpus: Whether to include results per corpus.
        strategy: Strategy for date range matching.
        date_from: Start date for filtering (inclusive).
        date_to: End date for filtering (inclusive).
        no_combined_cache: If True, do not use combined caching for multiple corpora.

    Returns:
        A dictionary containing the timespan information.

    Raises:
        ValueError: If only one of date_from or date_to is provided.
    """
    if (date_from or date_to) and not (date_from and date_to):
        raise ValueError("When using 'date_from' or 'date_to', both need to be specified.")
    total_start = perf_counter()
    fetch_duration = 0.0
    cache_write_duration = 0.0
    calc_duration = 0.0

    g_config = _GRANULARITY[granularity]

    cached_data = []
    corpora_rest = corpora.copy()

    @dataclass
    class TimespanCache:
        prefixes: dict[str, str]
        corpus_checksum: str
        combined_key: str

    cache = None
    cache_enabled = ctx.common.cache

    if cache_enabled:
        # Check if whole query is cached
        combined_checksum = utils.get_hash(
            (granularity, strategy, combined, per_corpus, date_from, date_to, sorted(corpora))
        )
        cache_prefix = await caching.cache_prefix(ctx.cache)
        cache_combined_key = f"{cache_prefix}:timespan_{combined_checksum}"
        result = await ctx.cache.get(cache_combined_key)
        if result is not None:
            if ctx.common.debug:
                result.setdefault("debug", {})
                result["debug"]["cache_read"] = True
            return result

        # Look for per-corpus caches
        corpus_checksum = utils.get_hash((date_from, date_to, granularity, strategy))
        cache_prefixes = await caching.cache_prefix(ctx.cache, corpora)
        for c in corpora:
            cache_key = f"{cache_prefixes[c]}:timespan_{corpus_checksum}"
            corpus_cached_data = await ctx.cache.get(cache_key)
            if corpus_cached_data is not None:
                cached_data.extend(corpus_cached_data)
                corpora_rest.remove(c)

        cache = TimespanCache(prefixes=cache_prefixes, corpus_checksum=corpus_checksum, combined_key=cache_combined_key)

    if corpora_rest:
        bind_params: dict[str, Any] = {}
        corpus_placeholders = ", ".join(f":corpus_{i}" for i in range(len(corpora_rest)))
        for i, c in enumerate(corpora_rest):
            bind_params[f"corpus_{i}"] = c

        fromto = ""
        if strategy == params.StrategyValues.some_overlaps:
            if date_from and date_to:
                fromto = (
                    " AND ((datefrom >= :date_from AND dateto <= :date_to)"
                    " OR (datefrom <= :date_from AND dateto >= :date_to))"
                )
                bind_params["date_from"] = date_from
                bind_params["date_to"] = date_to
        elif strategy == params.StrategyValues.all_overlaps:
            if date_to:
                fromto = " AND datefrom <= :date_to"
                bind_params["date_to"] = date_to
            if date_from:
                fromto += " AND dateto >= :date_from"
                bind_params["date_from"] = date_from
        elif strategy == params.StrategyValues.strict:
            if date_from:
                fromto = " AND datefrom >= :date_from"
                bind_params["date_from"] = date_from
            if date_to:
                fromto += " AND dateto <= :date_to"
                bind_params["date_to"] = date_to

        # TODO: Skip grouping on corpus when we're only after the combined results.
        # We do the granularity truncation and summation in the DB query if we can (depending on strategy),
        # since it's much faster than doing it afterwards

        timedata_table = "timedata_date" if g_config.uses_date_table else "timedata"
        if strategy == params.StrategyValues.some_overlaps:
            # We need the full dates for this strategy, so no truncating of the results
            # We cast datefrom/dateto to CHAR to avoid issues with year zero (which we use to represent unknown dates)
            sql = text(
                f"SELECT corpus, CAST(datefrom AS CHAR) AS df, CAST(dateto AS CHAR) AS dt, tokens AS sum"
                f" FROM {timedata_table}"
                f" WHERE corpus IN ({corpus_placeholders})"
                f"{fromto}"
                f" ORDER BY NULL"  # Avoid implicit ordering in older MySQL versions
            )
        else:
            left_len = g_config.sql_left_len
            sql = text(
                f"SELECT corpus, LEFT(datefrom, {left_len}) AS df, LEFT(dateto, {left_len}) AS dt,"
                f" SUM(tokens) AS sum"
                f" FROM {timedata_table}"
                f" WHERE corpus IN ({corpus_placeholders})"
                f"{fromto}"
                f" GROUP BY corpus, df, dt ORDER BY NULL"
            )

        async with ctx.db.async_connection() as conn:
            try:
                fetch_start = perf_counter()
                query_result = await conn.execute(sql, bind_params)
                rows_result = query_result.mappings().all()
                rows = [dict(row) for row in rows_result]
                fetch_duration = perf_counter() - fetch_start
            except Exception:
                await conn.invalidate()
                raise
    else:
        rows = []
        fetch_duration = 0.0

    max_cache_rows = max(0, settings.TIMESPAN_CACHE_MAX_ROWS)
    if cache_enabled and max_cache_rows and len(rows) > max_cache_rows:
        cache_enabled = False
        logger.debug(
            "Skipping timespan cache writes for large response (rows=%d > limit=%d)",
            len(rows),
            max_cache_rows,
        )

    if cache_enabled and cache:
        cache_write_start = perf_counter()

        async def save_cache(corpus: str, data: list[Mapping[str, Any]]) -> None:
            cache_key = f"{cache.prefixes[corpus]}:timespan_{cache.corpus_checksum}"
            try:
                await ctx.cache.add(cache_key, data)
            except CacheError:
                pass

        corpus_data = await _run_timespan_cpu_bound(_group_rows_by_corpus, rows, row_count=len(rows))
        for corpus, data in corpus_data.items():
            await save_cache(corpus, data)
        cache_write_duration = perf_counter() - cache_write_start

    calc_start = perf_counter()
    result = await _run_timespan_cpu_bound(
        _calculate_timespan_from_rows,
        cached_data,
        rows,
        granularity,
        combined,
        per_corpus,
        strategy,
        row_count=len(cached_data) + len(rows),
    )
    calc_duration = perf_counter() - calc_start

    if cache_enabled and cache and not no_combined_cache:
        # Save cache for whole query
        try:
            await ctx.cache.add(cache.combined_key, result)
        except CacheError:
            pass
    phase_log_seconds = max(0.0, settings.TIMESPAN_PHASE_LOG_SECONDS)
    total_duration = perf_counter() - total_start
    if phase_log_seconds and total_duration >= phase_log_seconds:
        logger.warning(
            "Timespan phases total=%.3fs fetch=%.3fs cache_write=%.3fs calculate=%.3fs rows=%d cached_rows=%d",
            total_duration,
            fetch_duration,
            cache_write_duration,
            calc_duration,
            len(rows),
            len(cached_data),
        )

    return result


def _group_rows_by_corpus(rows: list[Mapping[str, Any]]) -> defaultdict[str, list[Mapping[str, Any]]]:
    """Group SQL rows by corpus for per-corpus cache writes.

    Returns:
        A mapping from corpus name to all rows for that corpus.
    """
    corpus_data: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        corpus_data[row["corpus"]].append(row)
    return corpus_data


def _calculate_timespan_from_rows(
    cached_data: list[Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
    granularity: GranularityValues,
    combined: bool,
    per_corpus: bool,
    strategy: params.StrategyValues,
) -> dict:
    """Calculate timespan output from cached and newly fetched rows.

    Returns:
        The final timespan response payload.
    """
    return timespan_calculator(
        itertools.chain(cached_data, rows),
        granularity=granularity,
        combined=combined,
        per_corpus=per_corpus,
        strategy=strategy,
    )


async def _run_timespan_cpu_bound(function: Any, *args: Any, row_count: int) -> Any:
    """Run CPU-heavy timespan work in process for large inputs, with thread fallback.

    Returns:
        The return value produced by `function`.
    """
    threshold = max(0, settings.TIMESPAN_PROCESS_THRESHOLD_ROWS)
    if threshold and row_count >= threshold:
        logger.debug("Offloading timespan CPU stage to process (rows=%d)", row_count)
        try:
            return await anyio.to_process.run_sync(function, *args)
        except Exception as error:
            logger.debug("Timespan process offload failed, falling back to thread: %r", error)

    return await anyio.to_thread.run_sync(function, *args)


def _calculate_series_sweepline(
    segments: list[tuple[int, int]],
    corpus_intervals: list[tuple[int, int, int]],
    granularity: GranularityValues,
) -> defaultdict[str, int]:
    """Calculate timeseries using a sweep-line over interval starts.

    This algorithm efficiently calculates the frequency for each time bucket defined by the segments, taking into
    account the intervals that may start and end within those buckets. It uses a Fenwick tree to keep track of the
    frequencies of intervals that have started but not yet ended as we sweep through the segments.

    It requires that the segments are monotonic (non-decreasing) in their start points, which is guaranteed by the
    way we generate segments from the sorted nodes.

    Args:
        segments: List of (start, end) tuples representing the time buckets to calculate frequency for.
        corpus_intervals: List of (start, end, frequency) tuples representing the intervals for the corpus.
        granularity: The granularity level for date adjustments.

    Returns:
        A mapping from start date key to frequency for one corpus.
    """
    data: defaultdict[str, int] = defaultdict(int)
    if not segments:
        return data
    if not corpus_intervals:
        for start, end in segments:
            if start:
                data[str(start)] = 0
            if end:
                data[str(_adjust_date(str(end), granularity))] = 0
        return data

    intervals_by_start = sorted(corpus_intervals, key=itemgetter(0))
    end_values = sorted({item[1] for item in corpus_intervals})
    end_index = {value: index + 1 for index, value in enumerate(end_values)}  # 1-based for Fenwick tree
    fenwick = [0] * (len(end_values) + 1)

    def fenwick_add(index: int, value: int) -> None:
        while index < len(fenwick):
            fenwick[index] += value
            index += index & -index

    def fenwick_prefix_sum(index: int) -> int:
        total = 0
        while index > 0:
            total += fenwick[index]
            index -= index & -index
        return total

    total_started_freq = 0
    next_interval = 0

    for start, end in segments:
        while next_interval < len(intervals_by_start) and intervals_by_start[next_interval][0] <= start:
            _, interval_end, interval_freq = intervals_by_start[next_interval]
            fenwick_add(end_index[interval_end], interval_freq)
            total_started_freq += interval_freq
            next_interval += 1

        if start:
            data[str(start)] = 0

        excluded_freq = fenwick_prefix_sum(bisect.bisect_left(end_values, end))
        data[str(start or "")] += total_started_freq - excluded_freq

        if end:
            data[str(_adjust_date(str(end), granularity))] = 0

    return data


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
        granularity: Granularity of result.
        combined: Whether to include combined results.
        per_corpus: Whether to include results per corpus.
        strategy: Strategy for date range matching.

    Returns:
        A dictionary containing the timespan information.
    """
    g_config = _GRANULARITY[granularity]
    digit_len = g_config.digit_len

    def shorten_date(date: str) -> int:
        """Return a shortened version of the date according to the granularity."""
        alt = 1 if len(date) % 2 else 0  # Handle years with three digits
        return int(date[: digit_len - alt])

    intervals: defaultdict[str, list[tuple[int, int, int]]] = defaultdict(list)
    nodes: defaultdict[str, set[tuple[str, int]]] = defaultdict(set)

    datemin = "00000101" if g_config.uses_date_table else "00000101000000"
    datemax = "99991231" if g_config.uses_date_table else "99991231235959"

    for row in timedata:
        corpus = row["corpus"]
        datefrom = _digits_only(row["df"])
        dateto = _digits_only(row["dt"])
        datefrom_short = shorten_date(datefrom) if datefrom else 0
        dateto_short = shorten_date(dateto) if dateto else 0

        if strategy == params.StrategyValues.some_overlaps:
            # Some overlaps permitted
            # (t1 >= t1' AND t2 <= t2') OR (t1 <= t1' AND t2 >= t2')
            if datefrom_short != dateto_short:
                if datefrom[digit_len:] != datemin[digit_len:]:
                    datefrom_short = _adjust_date(str(datefrom_short), granularity)

                if dateto[digit_len:] != datemax[digit_len:]:
                    dateto_short = _adjust_date(str(dateto_short), granularity, subtract=True)

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

        interval = (datefrom_short, dateto_short, int(row["sum"]))
        if combined:
            intervals["__combined__"].append(interval)
            nodes["__combined__"].add(("f", datefrom_short))
            nodes["__combined__"].add(("t", dateto_short))
        if per_corpus:
            intervals[corpus].append(interval)
            nodes[corpus].add(("f", datefrom_short))
            nodes[corpus].add(("t", dateto_short))

    corpusnodes = {k: sorted(v, key=lambda x: (x[1] or 0, x[0])) for k, v in nodes.items()}
    result: dict[str, Any] = {}
    if per_corpus:
        result["corpora"] = {}
    if combined:
        result["combined"] = {}

    for corpus, nodes_ in corpusnodes.items():
        segments = []  # List of (start, end) tuples representing the time buckets to calculate frequency for
        for start_marker, end_marker in itertools.pairwise(nodes_):
            if start_marker[0] == "t":
                start = _adjust_date(str(start_marker[1]), granularity) if start_marker[1] else 0
                if start == end_marker[1] and end_marker[0] == "f":
                    continue
            else:
                start = start_marker[1]

            end = (
                0
                if not end_marker[1]
                else end_marker[1]
                if end_marker[0] == "t"
                else _adjust_date(str(end_marker[1]), granularity, subtract=True)
            )
            segments.append((start, end))

        corpus_intervals = intervals[corpus]
        # Segments are generated from node boundaries sorted by date; therefore start points are monotonic
        data = _calculate_series_sweepline(segments, corpus_intervals, granularity)

        if combined and corpus == "__combined__":
            result["combined"] = data
        else:
            result["corpora"][corpus] = data

    return result

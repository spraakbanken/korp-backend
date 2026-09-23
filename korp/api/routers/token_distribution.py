"""Router for token distribution information."""

import bisect
import calendar
import functools
import itertools
import re
from collections import defaultdict
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from logging import getLogger
from operator import itemgetter
from time import perf_counter
from typing import Annotated, Any, Literal, TypeAlias

import anyio.to_process
import anyio.to_thread
from dateutil.relativedelta import relativedelta
from fastapi import APIRouter, Query
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from sqlalchemy import text

from korp import auth, caching, utils
from korp.api import params, schemas
from korp.api.params import DateValue, GranularityValues
from korp.api.requests import QueryRequestModel, RequestModel
from korp.config import settings
from korp.dependencies import CtxDep, QueryCtxDep
from korp.handler import APIValidationError, api_handler, docs_response
from korp.memcached import CacheError

router = APIRouter(tags=["Statistics"])
logger = getLogger(__name__)

TOKEN_DISTRIBUTION_DESCRIPTION = f"""Show the distribution of corpus tokens over time.

The route returns token counts grouped by time period. Use `granularity` to choose the boundary resolution: year, month,
day, hour, minute, or second. The response can include per-corpus series, one combined series for all selected corpora,
or both.

Each series is an array of period records. `start` and `end` are inclusive ISO 8601 boundaries. Yearly, monthly, and
daily periods use dates; hourly, minute, and second periods use date-times. Undated material is represented by a record
with `dated: false` and no boundaries. Adjacent granularity units with the same token count are combined into one
period.

Use `date_from` and `date_to` together to limit the date range.

### Time Matching Strategies

{params.TIME_STRATEGY_DESCRIPTION}

### Example

Show yearly token distribution for a corpus:

`/token-distribution?corpora=VIVILL&granularity=year`
"""

DateFromParam: TypeAlias = Annotated[
    DateValue | SkipJsonSchema[None],
    Query(
        description=(
            "Start date/time for filtering, inclusive. Must be used together with `date_to`. Accepted formats: "
            "YYYYMMDDHHMMSS, YYYYMMDD, YYYY-MM-DD HH:MM:SS, or YYYY-MM-DD."
        ),
        examples=["20200101000000", "2020-01-01"],
    ),
]

DateToParam: TypeAlias = Annotated[
    DateValue | SkipJsonSchema[None],
    Query(
        description=(
            "End date/time for filtering, inclusive. Must be used together with `date_from`. Accepted formats: "
            "YYYYMMDDHHMMSS, YYYYMMDD, YYYY-MM-DD HH:MM:SS, or YYYY-MM-DD."
        ),
        examples=["20201231235959", "2020-12-31"],
    ),
]


class TokenDistributionRequest(RequestModel):
    """Request for corpus token distribution data."""

    json_array_fields = frozenset({"corpora"})

    corpora: params.CorporaParam
    granularity: params.GranularityParam = GranularityValues.year
    include_combined: params.IncludeCombinedParam = True
    include_per_corpus: params.IncludePerCorpusParam = True
    strategy: params.StrategyParam = params.StrategyValues.some_overlaps
    date_from: DateFromParam = None
    date_to: DateToParam = None


class TokenDistributionQuery(QueryRequestModel, TokenDistributionRequest):
    """GET query for corpus token distribution data."""

    csv_fields = TokenDistributionRequest.json_array_fields


@dataclass(frozen=True, slots=True)
class ValidatedDateRange:
    """Date bounds that have passed token-distribution validation."""

    date_from: str | None
    date_to: str | None
    parsed_date_from: datetime | None
    parsed_date_to: datetime | None


def validate_date_range(date_from: str | None, date_to: str | None) -> ValidatedDateRange:
    """Validate paired and ordered date bounds.

    Returns:
        The validated date bounds.

    Raises:
        APIValidationError: If the date range is incomplete or ordered incorrectly.
    """
    if (date_from or date_to) and not (date_from and date_to):
        raise APIValidationError("When using 'date_from' or 'date_to', both need to be specified.")
    parsed_date_from = parsed_date_to = None
    if date_from and date_to:
        try:
            parsed_date_from = utils.strptime(re.sub(r"\D", "", date_from))
            parsed_date_to = utils.strptime(re.sub(r"\D", "", date_to))
        except ValueError as exc:
            raise APIValidationError("Invalid date range.") from exc
        if parsed_date_from > parsed_date_to:
            raise APIValidationError("'date_from' must be before or equal to 'date_to'.")
    return ValidatedDateRange(
        date_from=date_from,
        date_to=date_to,
        parsed_date_from=parsed_date_from,
        parsed_date_to=parsed_date_to,
    )


PeriodBoundary: TypeAlias = date | datetime


class TokenDistributionPeriod(schemas.ResponseModel):
    """Token count for a dated period."""

    start: PeriodBoundary = Field(..., description="Inclusive ISO 8601 start boundary.", examples=["2010-01-01"])
    end: PeriodBoundary = Field(..., description="Inclusive ISO 8601 end boundary.", examples=["2011-12-31"])
    tokens: int = Field(..., description="Number of corpus tokens in the period.", examples=[354])


class UndatedTokenDistributionPeriod(schemas.ResponseModel):
    """Token count for material without usable date information."""

    dated: Literal[False] = Field(..., description="Always false for undated material.")
    tokens: int = Field(..., description="Number of undated corpus tokens.", examples=[42])


TokenDistributionPeriodRecord: TypeAlias = TokenDistributionPeriod | UndatedTokenDistributionPeriod


@dataclass(frozen=True, slots=True)
class TokenPeriodData:
    """Internal token count for one period.

    Dated boundaries are inclusive compact integers at the selected granularity. Both boundaries are `None` for
    undated material.
    """

    start: int | None
    end: int | None
    tokens: int

    def __post_init__(self) -> None:
        """Reject half-dated or inverted internal periods.

        Raises:
            ValueError: If only one boundary is set or the boundaries are inverted.
        """
        if (self.start is None) != (self.end is None):
            raise ValueError("Period boundaries must both be set or both be None.")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("Period start must not be after period end.")

    @property
    def dated(self) -> bool:
        """Whether this period has dated boundaries."""
        return self.start is not None


@dataclass(slots=True)
class TokenDistributionData:
    """Internal token-distribution result shared by route consumers."""

    granularity: GranularityValues
    corpora: dict[str, list[TokenPeriodData]] | None
    combined: list[TokenPeriodData] | None
    debug: dict[str, Any] | None = None


class TokenDistributionResponse(schemas.CommonResponse):
    """Response model for `/token-distribution` route."""

    corpora: dict[str, list[TokenDistributionPeriodRecord]] | SkipJsonSchema[None] = Field(
        None,
        description=("Token-count periods keyed by corpus id. Omitted when `include_per_corpus=false`."),
        examples=[{"romi": [{"start": "2017-01-01", "end": "2017-12-31", "tokens": 15366}]}],
    )
    combined: list[TokenDistributionPeriodRecord] | SkipJsonSchema[None] = Field(
        None,
        description=(
            "Combined token counts per time period across all selected corpora. Omitted when `include_combined=false`."
        ),
        examples=[[{"start": "2017-01-01", "end": "2017-12-31", "tokens": 15366}]],
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


def _compact_datetime(value: int) -> datetime:
    """Parse an internal compact period boundary.

    Returns:
        The boundary at the start of its granularity unit.
    """
    raw = str(value)
    return utils.strptime("0" + raw if len(raw) % 2 else raw)


def _format_period_start(value: int, granularity: GranularityValues) -> str:
    """Format an internal period start as a canonical ISO 8601 boundary.

    Returns:
        The public start boundary.
    """
    boundary = _compact_datetime(value)
    if granularity in {GranularityValues.year, GranularityValues.month, GranularityValues.day}:
        return boundary.date().isoformat()
    return boundary.isoformat(timespec="seconds")


def _format_period_end(value: int, granularity: GranularityValues) -> str:
    """Format an internal inclusive period end as a canonical ISO 8601 boundary.

    Returns:
        The public inclusive end boundary.
    """
    boundary = _compact_datetime(value)
    if granularity == GranularityValues.year:
        return date(boundary.year, 12, 31).isoformat()
    if granularity == GranularityValues.month:
        return date(boundary.year, boundary.month, calendar.monthrange(boundary.year, boundary.month)[1]).isoformat()
    if granularity == GranularityValues.day:
        return boundary.date().isoformat()
    if granularity == GranularityValues.hour:
        boundary = boundary.replace(minute=59, second=59)
    elif granularity == GranularityValues.minute:
        boundary = boundary.replace(second=59)
    return boundary.isoformat(timespec="seconds")


def serialize_period_bounds(start: int, end: int, granularity: GranularityValues) -> dict[str, str]:
    """Serialize internal inclusive bounds as canonical public boundaries.

    Returns:
        Public `start` and `end` fields.
    """
    return {
        "start": _format_period_start(start, granularity),
        "end": _format_period_end(end, granularity),
    }


def shift_period_boundary(value: int, granularity: GranularityValues, *, subtract: bool = False) -> int:
    """Shift a compact internal boundary by one granularity unit.

    Returns:
        The shifted compact boundary.
    """
    return _adjust_date(str(value), granularity, subtract=subtract)


def _coalesce_token_periods(
    periods: Iterable[TokenPeriodData], granularity: GranularityValues
) -> list[TokenPeriodData]:
    """Combine contiguous dated periods with identical token counts.

    Returns:
        Periods with redundant boundaries removed.
    """
    result: list[TokenPeriodData] = []
    for period in periods:
        previous = result[-1] if result else None
        if (
            previous is not None
            and previous.dated
            and period.dated
            and previous.tokens == period.tokens
            and previous.end is not None
            and period.start == shift_period_boundary(previous.end, granularity)
        ):
            result[-1] = TokenPeriodData(start=previous.start, end=period.end, tokens=period.tokens)
        else:
            result.append(period)
    return result


def serialize_token_period(period: TokenPeriodData, granularity: GranularityValues) -> dict[str, Any]:
    """Serialize one internal token period for the public response.

    Returns:
        A dated or undated public token-period record.
    """
    if not period.dated:
        return {"dated": False, "tokens": period.tokens}
    assert period.start is not None
    assert period.end is not None
    return {
        **serialize_period_bounds(period.start, period.end, granularity),
        "tokens": period.tokens,
    }


def serialize_token_distribution(distribution: TokenDistributionData) -> dict[str, Any]:
    """Convert an internal token-distribution result to its public period-record representation.

    Returns:
        A token-distribution result containing period arrays.
    """
    result: dict[str, Any] = {}
    if distribution.corpora is not None:
        result["corpora"] = {
            corpus: [serialize_token_period(period, distribution.granularity) for period in periods]
            for corpus, periods in distribution.corpora.items()
        }
    if distribution.combined is not None:
        result["combined"] = [
            serialize_token_period(period, distribution.granularity) for period in distribution.combined
        ]
    if distribution.debug is not None:
        result["debug"] = distribution.debug
    return result


async def _token_distribution_stream(
    ctx: CtxDep,
    corpora: list[str],
    granularity: GranularityValues,
    include_combined: bool,
    include_per_corpus: bool,
    strategy: params.StrategyValues,
    date_range: ValidatedDateRange,
) -> AsyncIterator[dict]:
    """Calculate and stream token distribution data from validated parameters.

    This wraps `get_token_distribution` to provide an async iterator interface for streaming responses.

    Yields:
        A dictionary containing the token distribution information.
    """
    distribution = await get_token_distribution(
        ctx,
        corpora,
        granularity=granularity,
        include_combined=include_combined,
        include_per_corpus=include_per_corpus,
        strategy=strategy,
        validated_date_range=date_range,
    )
    yield serialize_token_distribution(distribution)


@router.get(
    "/token-distribution",
    response_model=None,
    responses=docs_response(TokenDistributionResponse, corpus_authorization=True),
    summary="Token Distribution",
    description=TOKEN_DISTRIBUTION_DESCRIPTION,
    operation_id="get_token_distribution",
)
@api_handler
async def token_distribution_get(
    ctx: QueryCtxDep,
    query: Annotated[TokenDistributionQuery, Query()],
) -> AsyncIterator[dict]:
    """Calculate token distribution information for corpora.

    Returns:
        The token-distribution result stream.
    """
    return await _token_distribution(ctx, query.to_request(TokenDistributionRequest))


@router.post(
    "/token-distribution",
    response_model=None,
    responses=docs_response(TokenDistributionResponse, corpus_authorization=True),
    summary="Token Distribution",
    description=TOKEN_DISTRIBUTION_DESCRIPTION,
    operation_id="post_token_distribution",
)
@api_handler
async def token_distribution_post(ctx: CtxDep, request: TokenDistributionRequest) -> AsyncIterator[dict]:
    """Calculate token distribution information for corpora.

    Returns:
        The token-distribution result stream.
    """
    return await _token_distribution(ctx, request)


async def _token_distribution(ctx: CtxDep, request: TokenDistributionRequest) -> AsyncIterator[dict]:
    """Calculate token distribution information for corpora.

    Args:
        ctx: The request context.
        request: Validated token-distribution request.

    Returns:
        An async iterator yielding the token distribution information.

    Raises:
        APIValidationError: When both `include_per_corpus` and `include_combined` are false.
    """
    if not request.include_per_corpus and not request.include_combined:
        raise APIValidationError("At least one of `include_per_corpus` and `include_combined` must be true.")

    corpora = request.corpora or []
    date_range = validate_date_range(request.date_from, request.date_to)
    await auth.check_authorization(corpora, ctx)
    return _token_distribution_stream(
        ctx,
        corpora,
        request.granularity,
        request.include_combined,
        request.include_per_corpus,
        request.strategy,
        date_range,
    )


async def get_token_distribution(
    ctx: CtxDep,
    corpora: list[str],
    granularity: GranularityValues = GranularityValues.year,
    include_combined: bool = True,
    include_per_corpus: bool = True,
    strategy: params.StrategyValues = params.StrategyValues.some_overlaps,
    date_from: str | None = None,
    date_to: str | None = None,
    no_combined_cache: bool = False,
    validated_date_range: ValidatedDateRange | None = None,
) -> TokenDistributionData:
    """Fetch, cache, and calculate token distribution data for selected corpora.

    Args:
        ctx: The request context.
        corpora: List of corpora.
        granularity: Granularity of result.
        include_combined: Whether to include combined results.
        include_per_corpus: Whether to include results per corpus.
        strategy: Strategy for date range matching.
        date_from: Start date for filtering (inclusive).
        date_to: End date for filtering (inclusive).
        no_combined_cache: If True, do not use combined caching for multiple corpora.
        validated_date_range: Date bounds already validated before streaming starts.

    Returns:
        The internal token-distribution result containing period records.

    """
    if validated_date_range is None:
        validated_date_range = validate_date_range(date_from, date_to)
    date_from = validated_date_range.date_from
    date_to = validated_date_range.date_to
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
            (granularity, strategy, include_combined, include_per_corpus, date_from, date_to, sorted(corpora))
        )
        cache_prefix = await caching.cache_prefix(ctx.cache)
        cache_combined_key = f"{cache_prefix}:timespan_{combined_checksum}"
        result = await ctx.cache.get(cache_combined_key)
        if result is not None:
            assert isinstance(result, TokenDistributionData)
            if ctx.common.debug:
                result.debug = {**(result.debug or {}), "cache_read": True}
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
            bind_params[f"corpus_{i}"] = c.upper()

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
                rows = []
                for row in rows_result:
                    normalized_row = dict(row)
                    normalized_row["corpus"] = utils.normalize_corpus_id(normalized_row["corpus"])
                    rows.append(normalized_row)
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
            "Skipping token distribution cache writes for large response (rows=%d > limit=%d)",
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

        corpus_data = await _run_token_distribution_cpu_bound(_group_rows_by_corpus, rows, row_count=len(rows))
        for corpus, data in corpus_data.items():
            await save_cache(corpus, data)
        cache_write_duration = perf_counter() - cache_write_start

    calc_start = perf_counter()
    result = await _run_token_distribution_cpu_bound(
        _calculate_token_distribution_from_rows,
        cached_data,
        rows,
        granularity,
        include_combined,
        include_per_corpus,
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
            "Token distribution phases total=%.3fs fetch=%.3fs cache_write=%.3fs calculate=%.3fs rows=%d "
            "cached_rows=%d",
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


def _calculate_token_distribution_from_rows(
    cached_data: list[Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
    granularity: GranularityValues,
    include_combined: bool,
    include_per_corpus: bool,
    strategy: params.StrategyValues,
) -> TokenDistributionData:
    """Calculate token distribution output from cached and newly fetched rows.

    Returns:
        The internal token-distribution result containing period records.
    """
    return build_token_distribution(
        itertools.chain(cached_data, rows),
        granularity=granularity,
        include_combined=include_combined,
        include_per_corpus=include_per_corpus,
        strategy=strategy,
    )


async def _run_token_distribution_cpu_bound(function: Any, *args: Any, row_count: int) -> Any:
    """Run CPU-heavy token distribution work in process for large inputs, with thread fallback.

    Returns:
        The return value produced by `function`.
    """
    threshold = max(0, settings.TIMESPAN_PROCESS_THRESHOLD_ROWS)
    if threshold and row_count >= threshold:
        logger.debug("Offloading token distribution CPU stage to process (rows=%d)", row_count)
        try:
            return await anyio.to_process.run_sync(function, *args)
        except Exception as error:
            logger.debug("Token distribution process offload failed, falling back to thread: %r", error)

    return await anyio.to_thread.run_sync(function, *args)


def _calculate_series_sweepline(
    segments: list[tuple[int, int]],
    corpus_intervals: list[tuple[int, int, int]],
) -> list[TokenPeriodData]:
    """Calculate timeseries using a sweep-line over interval starts.

    This algorithm efficiently calculates the frequency for each time bucket defined by the segments, taking into
    account the intervals that may start and end within those buckets. It uses a Fenwick tree to keep track of the
    frequencies of intervals that have started but not yet ended as we sweep through the segments.

    It requires that the segments are monotonic (non-decreasing) in their start points, which is guaranteed by the
    way we generate segments from the sorted nodes.

    Args:
        segments: List of (start, end) tuples representing the time buckets to calculate frequency for.
        corpus_intervals: List of (start, end, frequency) tuples representing the intervals for the corpus.

    Returns:
        Non-overlapping token periods for one corpus.
    """
    if not segments:
        return []

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
    periods: list[TokenPeriodData] = []
    undated_tokens = 0
    has_undated = False

    for start, end in segments:
        while next_interval < len(intervals_by_start) and intervals_by_start[next_interval][0] <= start:
            _, interval_end, interval_freq = intervals_by_start[next_interval]
            fenwick_add(end_index[interval_end], interval_freq)
            total_started_freq += interval_freq
            next_interval += 1

        excluded_freq = fenwick_prefix_sum(bisect.bisect_left(end_values, end))
        tokens = total_started_freq - excluded_freq
        if start and end:
            periods.append(TokenPeriodData(start=start, end=end, tokens=tokens))
        else:
            has_undated = True
            undated_tokens += tokens

    if has_undated:
        periods.append(TokenPeriodData(start=None, end=None, tokens=undated_tokens))
    return periods


def build_token_distribution(
    timedata: Iterable[Mapping],
    granularity: GranularityValues = GranularityValues.year,
    include_combined: bool = True,
    include_per_corpus: bool = True,
    strategy: params.StrategyValues = params.StrategyValues.some_overlaps,
) -> TokenDistributionData:
    """Aggregate corpus time intervals into token counts grouped by time period.

    Args:
        timedata: List of time data dictionaries with keys 'corpus', 'df' (datefrom), 'dt' (dateto), and 'sum' (token
            count).
        granularity: Granularity of result.
        include_combined: Whether to include combined results.
        include_per_corpus: Whether to include results per corpus.
        strategy: Strategy for date range matching.

    Returns:
        The internal token-distribution result containing period records.
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
        if include_combined:
            intervals["__combined__"].append(interval)
            nodes["__combined__"].add(("f", datefrom_short))
            nodes["__combined__"].add(("t", dateto_short))
        if include_per_corpus:
            intervals[corpus].append(interval)
            nodes[corpus].add(("f", datefrom_short))
            nodes[corpus].add(("t", dateto_short))

    corpusnodes = {k: sorted(v, key=lambda x: (x[1] or 0, x[0])) for k, v in nodes.items()}
    corpora_result: dict[str, list[TokenPeriodData]] | None = {} if include_per_corpus else None
    combined_result: list[TokenPeriodData] | None = [] if include_combined else None

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
        periods = _coalesce_token_periods(_calculate_series_sweepline(segments, corpus_intervals), granularity)

        if include_combined and corpus == "__combined__":
            combined_result = periods
        else:
            assert corpora_result is not None
            corpora_result[corpus] = periods

    return TokenDistributionData(
        granularity=granularity,
        corpora=corpora_result,
        combined=combined_result,
    )

"""Routes for counting word/attribute occurrences in corpora."""

import dataclasses
import itertools
import re
from collections import defaultdict
from collections.abc import AsyncGenerator, AsyncIterator, Collection, Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Annotated, Any, TypeAlias, cast

import anyio
from anyio import CapacityLimiter
from dateutil.relativedelta import relativedelta
from fastapi import APIRouter, Query
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

from korp import auth, caching, cqp, handler, utils
from korp.api import params, schemas
from korp.config import settings
from korp.dependencies import AbortDep, AbortSignal, CtxDep
from korp.handler import api_handler
from korp.memcached import CacheError, MemcachedSyncClient

from . import info, token_distribution

if TYPE_CHECKING:
    import anyio.abc

RELATIVE_MULTIPLIER = 1_000_000  # For relative frequencies per million tokens
DATEFROM = "text_datefrom"
TIMEFROM = "text_timefrom"
DATETO = "text_dateto"
TIMETO = "text_timeto"


router = APIRouter(tags=["Statistics"])

COUNT_DESCRIPTION = """Calculate frequencies for one or more attributes in the result of a CQP query.

The response contains absolute counts and relative frequencies. Relative frequencies are expressed as hits per one
million tokens.

Use `group_by` for positional attributes and `group_by_struct` for structural attributes. If neither is supplied, the
route groups by `word`. To count the value of a specific token in a multi-token query, mark that token as the CQP target
with `@`, for example `[pos = "JJ"] @[pos = "NN"]`.

Repeat the `cqp` parameter to run prequeries in sequence. Repeat the `subcqp` parameter to add subqueries over the final
main-query result. When subqueries are used, `combined` and each entry in `corpora` become arrays: the first item is the
main query result and the following items are the subquery results, each with a `cqp` field.

When `incremental=true`, progress keys such as `progress_corpora` and `progress_0` may be included before the final
statistics in the streamed JSON object.
"""

COUNT_ALL_DESCRIPTION = """Calculate frequencies for all tokens in the selected corpora, grouped by the requested
attributes.

This is the optimized variant to use when no CQP query is needed, for example when listing all part-of-speech values or
all word forms in a corpus. It uses the same grouping and formatting parameters as `/count`, except it does not accept
`cqp` or `subcqp`.

If neither `group_by` nor `group_by_struct` is supplied, the route groups by `word`.
"""

COUNT_TIME_DESCRIPTION = f"""Calculate the frequency of a query over time.

The response contains absolute counts and relative frequencies per time period. Relative frequencies are expressed as
hits per one million tokens for the corresponding time period.

Each data point covers the period from that key until the next key. For example, with yearly granularity, values for
`2010`, `2012`, `2013`, and `2016` describe 2010-2011, 2012, 2013-2015, and 2016 onward respectively. A value of `null`
means there is no corpus data for that period; `0` means data exists but the query had no hits.

### Time Matching Strategies

{params.TIME_STRATEGY_DESCRIPTION}

Repeat the `subcqp` parameter to add subqueries over the final main-query result. When subqueries are used, `combined`
and each entry in `corpora` become arrays: the first item is the main query result and the following items are the
subquery results, each with a `cqp` field.
"""

GroupByParam: TypeAlias = Annotated[
    list[str] | SkipJsonSchema[None],
    Query(
        description=(
            "Comma-separated list of positional attributes to group results by. Defaults to `word` if neither "
            "`group_by` nor `group_by_struct` is supplied."
        ),
        examples=[["word"], ["pos,lemma"]],
    ),
    BeforeValidator(utils.split_csv),
]

GroupByStructParam: TypeAlias = Annotated[
    list[str] | SkipJsonSchema[None],
    Query(
        description=(
            "Comma-separated list of structural attributes to group results by. The value at the first token of the "
            "match is used."
        ),
        examples=[["text_author"], ["text_author,text_title"]],
    ),
    BeforeValidator(utils.split_csv),
]

OffsetParam: TypeAlias = Annotated[
    int,
    Query(description="Number of result rows to skip after sorting by absolute frequency.", ge=0, examples=[0]),
]

LimitParam: TypeAlias = Annotated[
    int,
    Query(
        description="Maximum number of result rows to return after `offset`. Use 0 for no limit.",
        ge=0,
        examples=[25],
    ),
]

IgnoreCaseParam: TypeAlias = Annotated[
    list[str] | SkipJsonSchema[None],
    Query(
        description="Comma-separated list of attributes whose values should be lowercased before counting.",
        examples=[["word"], ["word,lemma"]],
    ),
    BeforeValidator(utils.split_csv),
]

SubCQPParam: TypeAlias = Annotated[
    list[str] | SkipJsonSchema[None],
    Query(
        description=(
            "CQP subqueries to perform over the final main-query result. Repeat the `subcqp` parameter to provide "
            "multiple subqueries."
        ),
        examples=[['[lex contains "tsunami..nn.1"]', '[lex contains "flodvåg..nn.1"]']],
    ),
]

RelativeToStructParam: TypeAlias = Annotated[
    list[str] | SkipJsonSchema[None],
    Query(
        description=(
            "Structural attributes to use as the denominator for relative frequencies instead of total corpus size. "
            "Every value must also be included in `group_by_struct`."
        ),
        examples=[["text_author"]],
    ),
    BeforeValidator(utils.split_csv),
]

StripPointerParam: TypeAlias = Annotated[
    list[str] | SkipJsonSchema[None],
    Query(
        description="Comma-separated list of attributes whose multi-word pointer suffixes should be stripped.",
        examples=[["sense"]],
    ),
    BeforeValidator(utils.split_csv),
]

TopParam: TypeAlias = Annotated[
    list[str] | SkipJsonSchema[None],
    Query(
        description=(
            "Comma-separated list of attributes for which only the first N values in a set should be counted. "
            "Use `attr:n`; if `:n` is omitted, N defaults to 1. Usually used together with `split`."
        ),
        examples=[["sense:3"], ["sense:3,lemma"]],
    ),
    BeforeValidator(utils.split_csv),
]


class FrequencySums(BaseModel):
    """Frequency sums for a statistics result."""

    absolute: int = Field(..., description="Absolute frequency sum.", examples=[598])
    relative: float = Field(..., description="Relative frequency sum.", examples=[13.765536])


class CountRow(BaseModel):
    """A grouped count row."""

    value: dict[str, str | list[str]] = Field(
        ...,
        description=(
            "Grouped attribute values. Positional attributes are arrays with one value per token in the match; "
            "structural attributes normally contain one value."
        ),
        examples=[{"word": ["run"], "pos": ["VB"]}, {"text_author": ["Söderberg, Hjalmar"]}],
    )
    absolute: int = Field(..., description="Absolute frequency.", examples=[598])
    relative: float = Field(..., description="Relative frequency per one million tokens.", examples=[13.765536])


class CountStatistics(BaseModel):
    """Statistics for one query or subquery."""

    rows: list[CountRow] = Field(..., description="Grouped frequency rows.")
    sums: FrequencySums = Field(..., description="Frequency sums over all returned rows.")
    cqp: str | SkipJsonSchema[None] = Field(
        None, description="Subquery CQP string. Included only for `subcqp` results."
    )


class CountResponse(schemas.CommonResponse):
    """Response model for `/count` route."""

    model_config = ConfigDict(extra="allow")

    corpora: dict[str, CountStatistics | list[CountStatistics]] = Field(
        ...,
        description=(
            "Statistics per corpus. Values are arrays when `subcqp` is used; otherwise each value is one statistics "
            "object."
        ),
    )
    combined: CountStatistics | list[CountStatistics] = Field(
        ...,
        description=(
            "Combined statistics for all corpora. This is an array when `subcqp` is used; otherwise it is one "
            "statistics object."
        ),
    )
    count: int = Field(
        ..., description="Total number of distinct grouped values before response slicing.", examples=[241]
    )
    progress_corpora: list[str] | SkipJsonSchema[None] = Field(
        None,
        description="Corpora that will produce incremental progress updates, included only when `incremental=true`.",
        examples=[["ROMI", "SUC3"]],
    )


class CountAllResponse(schemas.CommonResponse):
    """Response model for `/count_all` route."""

    model_config = ConfigDict(extra="allow")

    corpora: dict[str, CountStatistics] = Field(..., description="Statistics per corpus.")
    combined: CountStatistics = Field(..., description="Combined statistics for all corpora.")
    count: int = Field(
        ..., description="Total number of distinct grouped values before response slicing.", examples=[241]
    )
    progress_corpora: list[str] | SkipJsonSchema[None] = Field(
        None,
        description="Corpora that will produce incremental progress updates, included only when `incremental=true`.",
        examples=[["ROMI", "SUC3"]],
    )


class TimeStatistics(BaseModel):
    """Time-series statistics for one query or subquery."""

    absolute: dict[str, int | None] | int = Field(
        ...,
        description=(
            "Absolute frequencies per time period. A value of `null` means there is no corpus data for that period; "
            "`0` means data exists but the query had no hits."
        ),
        examples=[{"2017": 354, "2018": 115, "2019": None}],
    )
    relative: dict[str, float | None] | float = Field(
        ...,
        description=(
            "Relative frequencies per time period. A value of `null` means there is no corpus data for that period; "
            "`0` means data exists but the query had no hits."
        ),
        examples=[{"2017": 65.265, "2018": 87.521, "2019": None}],
    )
    sums: FrequencySums = Field(..., description="Frequency sums over the time series.")
    cqp: str | SkipJsonSchema[None] = Field(
        None, description="Subquery CQP string. Included only for `subcqp` results."
    )


class CountTimeResponse(schemas.CommonResponse):
    """Response model for `/count_time` route."""

    model_config = ConfigDict(extra="allow")

    corpora: dict[str, TimeStatistics | list[TimeStatistics]] | SkipJsonSchema[None] = Field(
        None,
        description=(
            "Time-series statistics per corpus. Omitted when `per_corpus=false`. Values are arrays when `subcqp` is "
            "used; otherwise each value is one statistics object."
        ),
    )
    combined: TimeStatistics | list[TimeStatistics] | SkipJsonSchema[None] = Field(
        None,
        description=(
            "Combined time-series statistics for all corpora. Omitted when `combined=false`. This is an array when "
            "`subcqp` is used; otherwise it is one statistics object."
        ),
    )
    progress_corpora: list[str] | SkipJsonSchema[None] = Field(
        None,
        description="Corpora that will produce incremental progress updates, included only when `incremental=true`.",
        examples=[["ROMI", "SUC3"]],
    )


@dataclass
class CountParameters:
    """Parameters for count query, parsed and validated."""

    corpora: list[str]
    cqp_query: list[str | list[str]]
    subcqp: list[str] = dataclasses.field(default_factory=list)
    group_by: list[tuple[str, bool]] = dataclasses.field(default_factory=list)
    within: dict[str, str | None] = dataclasses.field(default_factory=lambda: defaultdict(lambda: None))
    ignore_case: set[str] = dataclasses.field(default_factory=set)
    simple: bool = False
    relative_to_struct: list[tuple[str, bool]] = dataclasses.field(default_factory=list)
    split: set[str] = dataclasses.field(default_factory=set)
    strip_pointer: set[str] = dataclasses.field(default_factory=set)
    top: dict[str, int] = dataclasses.field(default_factory=dict)
    expand_prequeries: bool = True
    start: int = 0
    end: int = -1
    cut: int | None = None


async def parse_parameters(
    ctx: CtxDep,
    corpus: list[str],
    cqp_query: list[str] | None,
    subcqp: list[str] | None,
    group_by: list[str] | None,
    group_by_struct: list[str] | None,
    within: list[str] | None,
    default_within: str | None,
    cut: int | None,
    ignore_case: list[str] | None,
    relative_to_struct: list[str] | None,
    split: list[str] | None,
    strip_pointer: list[str] | None,
    top: list[str] | None,
    simple: bool,
    expand_prequeries: bool,
    offset: int,
    limit: int,
) -> CountParameters:
    """Parse and validate parameters for count query.

    Returns:
        A CountParameters instance with parsed parameters.

    Raises:
        ValueError: If any parameter is invalid.
    """
    await auth.check_authorization(corpus, ctx)

    group_by = sorted(set(group_by)) if group_by else []
    group_by_struct = sorted(set(group_by_struct)) if group_by_struct else []

    if not group_by and not group_by_struct:
        group_by = ["word"]

    group_by_combined = [(g, False) for g in group_by] + [(g, True) for g in group_by_struct]

    ignore_case_set = set(ignore_case) if ignore_case else set()

    within_dict = cqp.parse_within(within, default_within)

    relative_to_structs = sorted(set(relative_to_struct)) if relative_to_struct else []
    if not all(r in group_by_struct for r in relative_to_structs):
        raise ValueError("All 'relative_to_struct' values also need to be present in 'group_by_struct'.")

    relative_to = [(r, True) for r in relative_to_structs]

    tops = {}
    if top:
        for t in top:
            if ":" in t:
                tops[t.split(":")[0]] = int(t.split(":")[1])
            else:
                tops[t] = 1

    subcqp = subcqp or []
    cqp_combined: list[str | list[str]] = []
    if cqp_query:
        cqp_combined.extend(cqp_query)

    if len(cqp_combined) > 1 and expand_prequeries and not all(within_dict[c] for c in corpus):
        raise ValueError("Multiple CQP queries requires 'within' or 'expand_prequeries=false'")

    if subcqp:
        cqp_combined.append(subcqp)

    if cqp_combined == ["[]"]:
        simple = True

    return CountParameters(
        corpora=corpus,
        cqp_query=cqp_combined,
        subcqp=subcqp,
        group_by=group_by_combined,
        within=within_dict,
        ignore_case=ignore_case_set,
        simple=simple,
        relative_to_struct=relative_to,
        split=set(split) if split else set(),
        strip_pointer=set(strip_pointer) if strip_pointer else set(),
        top=tops,
        expand_prequeries=expand_prequeries,
        start=offset,
        end=-1 if limit == 0 else offset + limit - 1,
        cut=cut,
    )


def _strip_pointer(tok: str) -> str:
    """Strip multi-word pointer suffix from a token.

    Args:
        tok: Token string, possibly with a pointer suffix like "word:123".

    Returns:
        Token with pointer suffix removed if present.
    """
    if ":" in tok:
        base, ptr = tok.rsplit(":", 1)
        if ptr.isnumeric():
            return base
    return tok


def _parse_ngram_groups(
    ngram_groups: list[str],
    group_by: list[tuple[str, bool]],
    split: set[str],
    strip_pointer: set[str],
    top: dict[str, int],
) -> list[tuple[tuple[str, ...], ...]]:
    """Parse ngram groups into expanded ngram tuples.

    Args:
        ngram_groups: Raw ngram group strings from count output.
        group_by: List of (attribute, is_struct) tuples.
        split: Attributes to split on.
        strip_pointer: Attributes to strip pointers from.
        top: Dict of attribute to top-N limit.

    Returns:
        List of ngram tuple collections, one per group.
    """
    all_ngrams = []

    for i, ngram in enumerate(ngram_groups):
        strip_ptrs = group_by[i][0] in strip_pointer

        # Split value sets and treat each value as a hit
        if group_by[i][0] in split:
            tokens = [t + "|" for t in ngram.split("| ")]  # We can't split on just space due to spaces in annotations
            tokens[-1] = tokens[-1][:-1]
            if group_by[i][0] in top:
                split_tokens = [
                    [x for x in token.split("|") if x][: top[group_by[i][0]]] if token != "|" else [""]
                    for token in tokens
                ]
            else:
                split_tokens = [[x for x in token.split("|") if x] if token != "|" else [""] for token in tokens]

            # Strip multi-word pointers if requested
            if strip_ptrs:
                for j in range(len(split_tokens)):
                    split_tokens[j] = [_strip_pointer(t) for t in split_tokens[j]]

            ngrams = tuple(itertools.product(*split_tokens))
        else:
            ngrams = (tuple(ngram.split(" ")),) if not group_by[i][1] else ((ngram,),)

        all_ngrams.append(ngrams)

    return all_ngrams


def _accumulate_ngram_stats(
    cross: list[tuple],
    freq: int,
    corpus_stats: list[dict],
    total_stats: list[dict],
    query_no: int,
    corpus_size: int,
    relative_to_struct: list[tuple[str, bool]],
    relative_to_pos: list[int],
    relative_to_freqs: dict | None,
    corpus: str,
) -> None:
    """Accumulate frequency statistics for ngrams.

    Args:
        cross: Cross product of ngram combinations.
        freq: Frequency count for this line.
        corpus_stats: Per-corpus statistics accumulator.
        total_stats: Combined statistics accumulator.
        query_no: Current query/subquery index.
        corpus_size: Size of the corpus.
        relative_to_struct: Structural attributes for relative calculation.
        relative_to_pos: Positions of relative_to_struct in group_by.
        relative_to_freqs: Pre-computed frequencies for relative calculation.
        corpus: Current corpus name.
    """
    cs_rows = corpus_stats[query_no]["rows"]
    cs_sums = corpus_stats[query_no]["sums"]
    ts_rows = total_stats[query_no]["rows"]
    ts_sums = total_stats[query_no]["sums"]

    for ngram in cross:
        cs_rows[ngram]["absolute"] += freq
        cs_sums["absolute"] += freq
        ts_rows[ngram]["absolute"] += freq
        ts_sums["absolute"] += freq

        if relative_to_struct and relative_to_freqs:
            # Only use the first token of each relative_to_struct attribute
            relativeto_ngram = tuple(ngram[pos][0:1] for pos in relative_to_pos)
            corpus_rel = freq / relative_to_freqs["corpora"][corpus][relativeto_ngram] * RELATIVE_MULTIPLIER
            cs_rows[ngram]["relative"] += corpus_rel
            cs_sums["relative"] += corpus_rel
            ts_rows[ngram]["relative"] += freq / relative_to_freqs["combined"][relativeto_ngram] * RELATIVE_MULTIPLIER
        else:
            rel = freq / corpus_size * RELATIVE_MULTIPLIER
            cs_rows[ngram]["relative"] += rel
            cs_sums["relative"] += rel


def _rows_to_list(rows: dict, group_by: list[tuple[str, bool]]) -> list[dict]:
    """Convert ngram-keyed row dict to a list of result dicts.

    Args:
        rows: Dict mapping ngram tuples to stat dicts (absolute/relative).
        group_by: List of (attribute, is_struct) tuples.

    Returns:
        List of dicts with "value" key and stat values.
    """
    return [{"value": {key[0]: ngram[i] for i, key in enumerate(group_by)}, **vals} for ngram, vals in rows.items()]


def _finalize_count_results(
    result: dict[str, Any],
    total_stats: list[dict],
    corpora: list[str],
    group_by: list[tuple[str, bool]],
    subcqp: list[str],
    relative_to_struct: list[tuple[str, bool]],
    total_size: int,
    start: int,
    end: int,
) -> None:
    """Finalize count results by calculating relative frequencies and formatting output.

    Args:
        result: Result dictionary to update in place.
        total_stats: Combined statistics.
        corpora: List of corpus names.
        group_by: List of (attribute, is_struct) tuples.
        subcqp: List of subqueries.
        relative_to_struct: Structural attributes for relative calculation.
        total_size: Total size across all corpora.
        start: Start index for result slicing.
        end: End index for result slicing. Use -1 for no upper bound.
    """
    for query_no in range(len(subcqp) + 1):
        slice_end = None if end == -1 else end + 1
        if start > 0 or (end > -1 and len(total_stats[query_no]["rows"]) > (end - start) + 1):
            # Only a selected range of results requested
            total_stats[query_no]["rows"] = dict(
                sorted(total_stats[query_no]["rows"].items(), key=lambda x: x[1]["absolute"], reverse=True)[
                    start:slice_end
                ]
            )

            for c in corpora:
                result["corpora"][c][query_no]["rows"] = {
                    k: v
                    for k, v in result["corpora"][c][query_no]["rows"].items()
                    if k in total_stats[query_no]["rows"]
                }

        if not relative_to_struct:
            for ngram, vals in total_stats[query_no]["rows"].items():
                total_stats[query_no]["rows"][ngram]["relative"] = (
                    vals["absolute"] / float(total_size) * RELATIVE_MULTIPLIER
                )

        for c in corpora:
            result["corpora"][c][query_no]["rows"] = _rows_to_list(result["corpora"][c][query_no]["rows"], group_by)

        total_stats[query_no]["sums"]["relative"] = (
            total_stats[query_no]["sums"]["absolute"] / float(total_size) * RELATIVE_MULTIPLIER
            if total_size > 0
            else 0.0
        )

        if subcqp and query_no > 0:
            total_stats[query_no]["cqp"] = subcqp[query_no - 1]

        total_stats[query_no]["rows"] = _rows_to_list(total_stats[query_no]["rows"], group_by)


async def perform_count(
    count_params: CountParameters,
    ctx: CtxDep,
    abort_signal: AbortSignal | None,
) -> AsyncGenerator[dict]:
    """Perform the count query based on the given parameters.

    This is a helper function called by route handlers.

    Args:
        count_params: Parsed count query parameters.
        ctx: The request context.
        abort_signal: Event to signal abortion of the query.

    Yields:
        Count results as dictionaries.

    Raises:
        ValueError: If there is an error parsing the results.
    """
    incremental = ctx.common.incremental
    corpora = count_params.corpora
    cqp_combined = count_params.cqp_query
    subcqp = count_params.subcqp
    group_by = count_params.group_by
    within = count_params.within
    ignore_case = count_params.ignore_case
    simple = count_params.simple
    relative_to_struct = count_params.relative_to_struct
    split = count_params.split
    strip_pointer = count_params.strip_pointer
    top = count_params.top
    expand_prequeries = count_params.expand_prequeries
    start = count_params.start
    end = count_params.end

    result: dict[str, Any] = {"corpora": {}}
    debug = {}
    zero_hits: set[str] = set()
    read_from_cache = 0
    count_state = utils.Namespace()
    count_state.total_size = 0

    if ctx.common.cache:
        # Use cache to skip corpora with zero hits
        memcached_keys = {}
        cache_prefixes = await caching.cache_prefix(ctx.cache, corpora)
        for c in corpora:
            corpus_checksum = utils.get_hash(
                (cqp_combined, group_by, within[c], sorted(ignore_case), expand_prequeries)
            )
            memcached_keys[f"{cache_prefixes[c]}:count_size_{corpus_checksum}"] = c

        cached_size = await ctx.cache.get_many(memcached_keys.keys())
        for key in cached_size:
            nr_hits = cached_size[key][0]
            read_from_cache += 1
            if nr_hits == 0:
                zero_hits.add(memcached_keys[key])
                count_state.total_size += cached_size[key][1]

        if ctx.common.debug:
            debug["cache_coverage"] = f"{read_from_cache}/{len(corpora)}"

    total_stats = [
        {"rows": defaultdict(lambda: {"absolute": 0, "relative": 0.0}), "sums": {"absolute": 0, "relative": 0.0}}
        for _ in range(len(subcqp) + 1)
    ]

    # If relative_to_struct is used, perform a separate count to get frequencies for calculating relative numbers
    relative_to_freqs = {}
    if relative_to_struct:
        relative_parameters = CountParameters(
            cqp_query=["[]"],
            corpora=corpora,
            group_by=relative_to_struct,  # Group by struct
            split=split,
        )

        relative_to_result = await utils.async_generator_to_dict(perform_count(relative_parameters, ctx, abort_signal))
        relative_to_freqs = {"combined": {}, "corpora": defaultdict(dict)}

        for row in relative_to_result["combined"]["rows"]:
            relative_to_freqs["combined"][tuple(v for k, v in sorted(row["value"].items()))] = row["absolute"]

        for c in relative_to_result["corpora"]:
            for row in relative_to_result["corpora"][c]["rows"]:
                relative_to_freqs["corpora"][c][tuple(v for k, v in sorted(row["value"].items()))] = row["absolute"]

    count_function = count_query_worker if not simple else count_query_worker_simple

    count_state.progress_count = 0
    if incremental:
        # Initial yield to indicate which corpora will be processed
        yield {"progress_corpora": [c for c in corpora if c not in zero_hits]}

    # Add zero-hit corpora to result
    for c in zero_hits:
        result["corpora"][c] = [{"rows": {}, "sums": {"absolute": 0, "relative": 0.0}} for _ in range(len(subcqp) + 1)]
        for i in range(len(subcqp)):
            result["corpora"][c][i + 1]["cqp"] = subcqp[i]

    if abort_signal and abort_signal.is_set():
        return

    # Calculate which positions in group_by correspond to relative_to_struct for later use in workers
    relative_to_pos = [i for i, g in enumerate(group_by) if relative_to_struct and g in relative_to_struct]

    limiter = CapacityLimiter(settings.PARALLEL_THREADS)
    send, receive = anyio.create_memory_object_stream(0)

    async def _worker(corpus: str, send_channel: anyio.abc.ObjectSendStream) -> None:
        """Worker function to run count query in thread.

        Args:
            corpus: The corpus to query.
            send_channel: The channel to send results back.

        Raises:
            CQPError: If the CQP query fails.
        """
        async with send_channel:  # Closes the channel when done
            if abort_signal and abort_signal.is_set():
                return
            try:
                lines, nr_hits, corpus_size = await anyio.to_thread.run_sync(  # type: ignore
                    partial(  # Use partial to be able to pass keyword arguments
                        count_function,
                        ctx=ctx,
                        corpus=corpus,
                        cqp_query=cqp_combined,
                        group_by=group_by,
                        within=within[corpus],
                        ignore_case=ignore_case,
                        expand_prequeries=expand_prequeries,
                        use_cache=ctx.common.cache,
                        cache_max=settings.CACHE_MAX_STATS,
                        abort_signal=abort_signal,
                    ),
                    limiter=limiter,
                )
            except Exception as e:
                raise cqp.CQPError(str(e)) from e

            await send_channel.send((corpus, lines, nr_hits, corpus_size))

    async with anyio.create_task_group() as tg:
        for c in corpora:
            if c not in zero_hits:
                tg.start_soon(_worker, c, send.clone())

        await send.aclose()  # Close the original send channel

        async for c, lines, _nr_hits, corpus_size in receive:
            if abort_signal and abort_signal.is_set():
                tg.cancel_scope.cancel()
                return

            count_state.total_size += corpus_size
            corpus_stats = [
                {
                    "rows": defaultdict(lambda: {"absolute": 0, "relative": 0.0}),
                    "sums": {"absolute": 0, "relative": 0.0},
                }
                for _ in range(len(subcqp) + 1)
            ]

            query_no = 0
            for line in lines:
                if line == cqp.END_OF_LINE:
                    # EOL means the start of a new subcqp result
                    query_no += 1
                    if subcqp:
                        corpus_stats[query_no]["cqp"] = subcqp[query_no - 1]
                    continue
                freq, ngram = line.lstrip().split(" ", 1)

                ngram_groups = ngram.split("\t") if len(group_by) > 1 else [ngram]

                # Sanity check: ngram_groups must match expected number of group_by columns
                if len(ngram_groups) != len(group_by):
                    raise ValueError(
                        f"Error parsing result for corpus '{c}'. This is most likely due to a structural "
                        "attribute containing tabs, which is not supported."
                    )

                all_ngrams = _parse_ngram_groups(ngram_groups, group_by, split, strip_pointer, top)
                cross = list(itertools.product(*all_ngrams))

                _accumulate_ngram_stats(
                    cross=cross,
                    freq=int(freq),
                    corpus_stats=corpus_stats,
                    total_stats=total_stats,
                    query_no=query_no,
                    corpus_size=corpus_size,
                    relative_to_struct=relative_to_struct,
                    relative_to_pos=relative_to_pos,
                    relative_to_freqs=relative_to_freqs if relative_to_struct else None,
                    corpus=c,
                )

            result["corpora"][c] = corpus_stats

            if incremental:
                yield {f"progress_{count_state.progress_count}": c}
                count_state.progress_count += 1

    result["count"] = len(total_stats[0]["rows"])

    if abort_signal and abort_signal.is_set():
        return

    _finalize_count_results(
        result=result,
        total_stats=total_stats,
        corpora=corpora,
        group_by=group_by,
        subcqp=subcqp,
        relative_to_struct=relative_to_struct,
        total_size=count_state.total_size,
        start=start,
        end=end,
    )

    result["combined"] = total_stats if len(total_stats) > 1 else total_stats[0]

    if not subcqp:
        for c in corpora:
            result["corpora"][c] = result["corpora"][c][0]

    if ctx.common.debug:
        debug.update({"cqp": cqp_combined, "simple": simple})
        result["debug"] = debug

    yield result


@router.get(
    "/count",
    response_model=None,
    responses=handler.docs_response(CountResponse),
    summary="Statistics",
    description=COUNT_DESCRIPTION,
)
@router.post("/count", response_model=None, include_in_schema=False)
@api_handler
async def count(
    ctx: CtxDep,
    corpus: params.CorpusParam,
    cqp_query: params.CQPParam,
    subcqp: SubCQPParam = None,
    group_by: GroupByParam = None,
    group_by_struct: GroupByStructParam = None,
    within: params.WithinParam = None,
    default_within: params.DefaultWithinParam = None,
    # cut: int | None = None,
    offset: OffsetParam = 0,
    limit: LimitParam = 0,
    ignore_case: IgnoreCaseParam = None,
    relative_to_struct: RelativeToStructParam = None,
    split: params.SplitParam = None,
    strip_pointer: StripPointerParam = None,
    top: TopParam = None,
    expand_prequeries: params.ExpandPrequeriesParam = True,
    abort_signal: AbortDep = None,
) -> AsyncIterator[dict]:
    """Perform a CQP query and return a count of the given words/attributes.

    Yields:
        Count results as dictionaries.
    """
    count_params = await parse_parameters(
        ctx=ctx,
        corpus=corpus,
        cqp_query=cqp_query,
        subcqp=subcqp,
        group_by=group_by,
        group_by_struct=group_by_struct,
        within=within,
        default_within=default_within,
        cut=None,
        ignore_case=ignore_case,
        relative_to_struct=relative_to_struct,
        split=split,
        strip_pointer=strip_pointer,
        top=top,
        simple=False,
        expand_prequeries=expand_prequeries,
        offset=offset,
        limit=limit,
    )

    async for item in perform_count(count_params, ctx, abort_signal):
        yield item


@router.get(
    "/count_all",
    response_model=None,
    responses=handler.docs_response(CountAllResponse),
    summary="Complete Statistics",
    description=COUNT_ALL_DESCRIPTION,
)
@router.post("/count_all", response_model=None, include_in_schema=False)
@api_handler
async def count_all(
    ctx: CtxDep,
    corpus: params.CorpusParam,
    group_by: GroupByParam = None,
    group_by_struct: GroupByStructParam = None,
    within: params.WithinParam = None,
    default_within: params.DefaultWithinParam = None,
    # cut: int | None = None,
    offset: OffsetParam = 0,
    limit: LimitParam = 0,
    ignore_case: IgnoreCaseParam = None,
    relative_to_struct: RelativeToStructParam = None,
    split: params.SplitParam = None,
    strip_pointer: StripPointerParam = None,
    top: TopParam = None,
    expand_prequeries: params.ExpandPrequeriesParam = True,
    abort_signal: AbortDep = None,
) -> AsyncIterator[dict]:
    """Like `/count` but for every single value of the given attributes.

    Yields:
        Count results as dictionaries.
    """
    count_params = await parse_parameters(
        ctx=ctx,
        corpus=corpus,
        cqp_query=["[]"],
        subcqp=None,
        group_by=group_by,
        group_by_struct=group_by_struct,
        within=within,
        default_within=default_within,
        cut=None,
        ignore_case=ignore_case,
        relative_to_struct=relative_to_struct,
        split=split,
        strip_pointer=strip_pointer,
        top=top,
        simple=True,
        expand_prequeries=expand_prequeries,
        offset=offset,
        limit=limit,
    )

    async for item in perform_count(count_params, ctx, abort_signal):
        yield item


DateFromParam: TypeAlias = Annotated[
    str | SkipJsonSchema[None],
    Query(
        description=(
            "Start date/time for filtering, inclusive. Must be used together with `date_to`. Accepted formats: "
            "YYYYMMDDHHMMSS, YYYYMMDD, YYYY-MM-DD HH:MM:SS, or YYYY-MM-DD."
        ),
        pattern=r"^(\d{8}(\d{6})?|\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?)$",
        examples=["20200101000000", "2020-01-01"],
    ),
]
DateToParam: TypeAlias = Annotated[
    str | SkipJsonSchema[None],
    Query(
        description=(
            "End date/time for filtering, inclusive. Must be used together with `date_from`. Accepted formats: "
            "YYYYMMDDHHMMSS, YYYYMMDD, YYYY-MM-DD HH:MM:SS, or YYYY-MM-DD."
        ),
        pattern=r"^(\d{8}(\d{6})?|\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?)$",
        examples=["20201231235959", "2020-12-31"],
    ),
]


@router.get(
    "/count_time",
    response_model=None,
    responses=handler.docs_response(CountTimeResponse),
    summary="Statistics Over Time",
    description=COUNT_TIME_DESCRIPTION,
)
@router.post("/count_time", response_model=None, include_in_schema=False)
@api_handler
async def count_time(
    ctx: CtxDep,
    corpus: params.CorpusParam,
    cqp_query: params.CQPParam,
    subcqp: SubCQPParam = None,
    within: params.WithinParam = None,
    default_within: params.DefaultWithinParam = None,
    # cut: int | None = None,
    offset: OffsetParam = 0,
    limit: LimitParam = 0,
    ignore_case: IgnoreCaseParam = None,
    relative_to_struct: RelativeToStructParam = None,
    split: params.SplitParam = None,
    strip_pointer: StripPointerParam = None,
    top: TopParam = None,
    expand_prequeries: params.ExpandPrequeriesParam = True,
    granularity: params.GranularityParam = params.GranularityValues.year,
    date_from: DateFromParam = None,
    date_to: DateToParam = None,
    strategy: params.StrategyParam = params.StrategyValues.some_overlaps,
    combined: params.CombinedParam = True,
    per_corpus: params.PerCorpusParam = True,
    abort_signal: AbortDep = None,
) -> AsyncIterator[dict]:
    """Count occurrences per time period.

    Yields:
        Count results as dictionaries.

    Raises:
        ValueError: If parameters are invalid or if the date range is too large for the selected granularity.
    """
    count_params = await parse_parameters(
        ctx=ctx,
        corpus=corpus,
        cqp_query=cqp_query,
        subcqp=subcqp,
        group_by=None,
        group_by_struct=None,
        within=within,
        default_within=default_within,
        cut=None,
        ignore_case=ignore_case,
        relative_to_struct=relative_to_struct,
        split=split,
        strip_pointer=strip_pointer,
        top=top,
        simple=True,
        expand_prequeries=expand_prequeries,
        offset=offset,
        limit=limit,
    )

    incremental = ctx.common.incremental

    # Check that we have a suitable date range for the selected granularity
    df = None
    dt = None

    if (date_from or date_to) and not (date_from and date_to):
        raise ValueError("When using 'date_from' or 'date_to', both need to be specified.")

    result = {}
    if per_corpus:
        result["corpora"] = {}
    if ctx.common.debug:
        result["debug"] = {"cqp": count_params.cqp_query}

    # Get date range of selected corpora
    corpus_data = await info.get_corpus_info(ctx=ctx, corpora=count_params.corpora, no_combined_cache=True)
    corpora_copy = count_params.corpora.copy()

    def _parse_corpus_date(date_str: str) -> datetime:
        return utils.strptime(re.sub(r"\D", "", date_str))

    if date_from and date_to:
        date_from = re.sub(r"\D", "", date_from)
        date_to = re.sub(r"\D", "", date_to)
        df = utils.strptime(date_from)
        dt = utils.strptime(date_to)

        # Remove corpora not within selected date span
        for c in corpus_data["corpora"]:
            first_date = corpus_data["corpora"][c]["info"].get("FirstDate")
            last_date = corpus_data["corpora"][c]["info"].get("LastDate")
            if first_date and last_date:
                first_date = _parse_corpus_date(first_date)
                last_date = _parse_corpus_date(last_date)

                if not (first_date <= dt and last_date >= df):
                    count_params.corpora.remove(c)
    else:
        # If no date range was provided, use whole date range of the selected corpora
        for c in corpus_data["corpora"]:
            first_date = corpus_data["corpora"][c]["info"].get("FirstDate")
            last_date = corpus_data["corpora"][c]["info"].get("LastDate")
            if first_date and last_date:
                first_date = _parse_corpus_date(first_date)
                last_date = _parse_corpus_date(last_date)

                if not df or first_date < df:
                    df = first_date
                if not dt or last_date > dt:
                    dt = last_date

    if df and dt:
        max_points = 3600

        granularity_units = {
            granularity.year: "years",
            granularity.month: "months",
            granularity.day: "days",
            granularity.hour: "hours",
            granularity.minute: "minutes",
            granularity.second: "seconds",
        }
        add = relativedelta(**{granularity_units[granularity]: max_points})  # type: ignore

        if dt > (df + add):
            raise ValueError(
                "The date range is too large for the selected granularity. Use 'to' and 'from' to limit the range."
            )

    if granularity in {granularity.hour, granularity.minute, granularity.second}:
        group_by = [(v, True) for v in (DATEFROM, TIMEFROM, DATETO, TIMETO)]
    else:
        group_by = [(v, True) for v in (DATEFROM, DATETO)]

    if per_corpus:
        # Add zero values for the corpora we removed because of the selected date span
        for c in set(corpora_copy).difference(set(count_params.corpora)):
            result["corpora"][c] = [
                {"absolute": 0, "relative": 0.0, "sums": {"absolute": 0, "relative": 0.0}}
                for _ in range(len(count_params.subcqp) + 1)
            ]
            for i, c2 in enumerate(result["corpora"][c][1:]):
                c2["cqp"] = count_params.subcqp[i]

            if not count_params.subcqp:
                result["corpora"][c] = result["corpora"][c][0]

    # Add zero values for the combined results if no corpora are within the selected date span
    if combined and not count_params.corpora:
        result["combined"] = [
            {"absolute": 0, "relative": 0.0, "sums": {"absolute": 0, "relative": 0.0}}
            for _ in range(len(count_params.subcqp) + 1)
        ]
        for i, c in enumerate(result["combined"][1:]):
            c["cqp"] = count_params.subcqp[i]

        if not count_params.subcqp:
            result["combined"] = result["combined"][0]

        yield result
        return

    ns = utils.Namespace()
    total_rows = [[] for _ in range(len(count_params.subcqp) + 1)]
    ns.total_size = 0

    ns.progress_count = 0
    if incremental:
        yield {"progress_corpora": count_params.corpora}

    limiter = CapacityLimiter(settings.PARALLEL_THREADS)
    send, receive = anyio.create_memory_object_stream(0)

    async def _worker(corpus: str, send_channel: anyio.abc.ObjectSendStream) -> None:  # type: ignore
        """Worker function to run count query in thread.

        Args:
            corpus: The corpus to query.
            send_channel: The channel to send results back.

        Raises:
            CQPError: If the CQP query fails.
        """
        async with send_channel:
            if abort_signal and abort_signal.is_set():
                return
            try:
                lines, _, corpus_size = await anyio.to_thread.run_sync(  # type: ignore
                    partial(  # Use partial to be able to pass keyword arguments
                        count_query_worker,
                        corpus=corpus,
                        cqp_query=count_params.cqp_query,
                        group_by=group_by,
                        within=count_params.within[corpus],
                        expand_prequeries=count_params.expand_prequeries,
                        use_cache=ctx.common.cache,
                        cache_max=settings.CACHE_MAX_STATS,
                        abort_signal=abort_signal,
                        ctx=ctx,
                    ),
                    limiter=limiter,
                )
            except Exception as e:
                if f"Can't find attribute ``{DATEFROM}''" in str(e):
                    # Corpus lacks date attributes required for count_time; treat as no rows
                    await send_channel.send((corpus, (), 0))
                    return
                raise cqp.CQPError(str(e)) from e

            await send_channel.send((corpus, lines, corpus_size))

    async with anyio.create_task_group() as tg:
        for c in count_params.corpora:
            tg.start_soon(_worker, c, send.clone())

        await send.aclose()  # Close the original send channel

        async for c, lines, corpus_size in receive:
            ns.total_size += corpus_size

            query_no = 0
            for line in lines:
                if line == cqp.END_OF_LINE:
                    query_no += 1
                    continue
                count, values = line.lstrip().split(" ", 1)
                values = values.strip(" ")
                if granularity in {granularity.hour, granularity.minute, granularity.second}:
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

                total_rows[query_no].append(
                    {"corpus": c, "df": datefrom + timefrom, "dt": dateto + timeto, "sum": int(count)}
                )

            if incremental:
                yield {f"progress_{ns.progress_count}": c}
                ns.progress_count += 1

    corpus_timedata = await token_distribution.get_timespan(
        ctx=ctx,
        corpora=count_params.corpora,
        granularity=granularity,
        date_from=date_from,
        date_to=date_to,
        strategy=strategy,
        no_combined_cache=True,
    )

    search_timedata = []
    search_timedata_combined = []
    for total_row in total_rows:
        temp = token_distribution.timespan_calculator(total_row, granularity=granularity, strategy=strategy)
        if per_corpus:
            search_timedata.append(temp["corpora"])
        if combined:
            search_timedata_combined.append(temp["combined"])

    if per_corpus:
        for c in count_params.corpora:
            corpus_stats = [
                {"absolute": defaultdict(int), "relative": defaultdict(float), "sums": {"absolute": 0, "relative": 0.0}}
                for _ in range(len(count_params.subcqp) + 1)
            ]

            basedates = {
                date: None if corpus_timedata["corpora"][c][date] == 0 else 0
                for date in corpus_timedata["corpora"].get(c, {})
            }

            for i, s in enumerate(search_timedata):
                prevdate = None
                for basedate in sorted(basedates):
                    if basedates[basedate] != prevdate:
                        corpus_stats[i]["absolute"][basedate] = basedates[basedate]
                        corpus_stats[i]["relative"][basedate] = basedates[basedate]
                    prevdate = basedates[basedate]

                for row in s.get(c, {}).items():
                    date, count = row
                    corpus_date_size = float(corpus_timedata["corpora"].get(c, {}).get(date, 0))
                    if corpus_date_size > 0.0:
                        corpus_stats[i]["absolute"][date] += count
                        corpus_stats[i]["relative"][date] += count / corpus_date_size * RELATIVE_MULTIPLIER
                        corpus_stats[i]["sums"]["absolute"] += count
                        corpus_stats[i]["sums"]["relative"] += count / corpus_date_size * RELATIVE_MULTIPLIER

                if count_params.subcqp and i > 0:
                    corpus_stats[i]["cqp"] = count_params.subcqp[i - 1]

            result["corpora"][c] = corpus_stats if len(corpus_stats) > 1 else corpus_stats[0]

    if combined:
        total_stats = [
            {"absolute": defaultdict(int), "relative": defaultdict(float), "sums": {"absolute": 0, "relative": 0.0}}
            for _ in range(len(count_params.subcqp) + 1)
        ]

        basedates = {
            date: None if corpus_timedata["combined"][date] == 0 else 0 for date in corpus_timedata.get("combined", {})
        }

        for i, s in enumerate(search_timedata_combined):
            prevdate = None
            for basedate in sorted(basedates):
                if basedates[basedate] != prevdate:
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
                            (count / combined_date_size * RELATIVE_MULTIPLIER) if combined_date_size else 0
                        )
                        total_stats[i]["sums"]["absolute"] += count

            total_stats[i]["sums"]["relative"] = (
                total_stats[i]["sums"]["absolute"] / float(ns.total_size) * RELATIVE_MULTIPLIER
                if ns.total_size > 0
                else 0.0
            )
            if count_params.subcqp and i > 0:
                total_stats[i]["cqp"] = count_params.subcqp[i - 1]

        result["combined"] = total_stats if len(total_stats) > 1 else total_stats[0]

    yield result


@dataclass
class _CountCacheKeys:
    """Cache keys for count query results."""

    data_key: str
    size_key: str


def _get_count_cache_keys(
    corpus: str,
    cqp_query: list[str | list[str]],
    group_by: list[tuple[str, bool]],
    within: str | None,
    ignore_case: Collection[str],
    expand_prequeries: bool,
    mc: MemcachedSyncClient,
) -> _CountCacheKeys:
    """Generate cache keys for count query.

    Args:
        corpus: The corpus name.
        cqp_query: The CQP query.
        group_by: Attributes to group by.
        within: The within context.
        ignore_case: Attributes to ignore case for.
        expand_prequeries: Whether to expand prequeries.
        mc: The memcached client to use for generating cache prefix.

    Returns:
        A _CountCacheKeys instance with data and size cache keys.
    """
    checksum = utils.get_hash((cqp_query, group_by, within, sorted(ignore_case), expand_prequeries))
    prefix = caching.cache_prefix_sync(mc, corpus)
    return _CountCacheKeys(
        data_key=f"{prefix}:count_data_{checksum}",
        size_key=f"{prefix}:count_size_{checksum}",
    )


def _check_count_cache(
    cache_keys: _CountCacheKeys,
    zero_hit_result: Iterable[str],
    mc: MemcachedSyncClient,
) -> tuple[Iterable[str], int, int] | None:
    """Check cache for count query results.

    Args:
        cache_keys: The cache keys to check.
        zero_hit_result: The result to return if cache indicates zero hits.
        mc: The memcached client to use for checking the cache.

    Returns:
        Cached result tuple (lines, hits, size) if found, None otherwise.
    """
    cached_size = mc.get(cache_keys.size_key)
    if cached_size is None:
        return None

    corpus_hits, corpus_size = cached_size
    if corpus_hits == 0:
        return zero_hit_result, corpus_hits, corpus_size

    cached_result = mc.get(cache_keys.data_key)
    if cached_result is not None:
        return cached_result, corpus_hits, corpus_size

    return None


def _save_count_cache(
    cache_keys: _CountCacheKeys,
    lines: Iterable[str],
    nr_hits: int,
    corpus_size: int,
    cache_max: int,
    mc: MemcachedSyncClient,
) -> tuple[str, ...]:
    """Save count query results to cache.

    Args:
        cache_keys: The cache keys to use.
        lines: The result lines to cache.
        nr_hits: Number of hits.
        corpus_size: Size of the corpus.
        cache_max: Maximum number of lines to cache.
        mc: The memcached client to use for saving the cache.

    Returns:
        The lines as a tuple (for consistent return type).
    """
    lines_list = list(lines) if not isinstance(lines, list) else lines
    mc.add(cache_keys.size_key, (nr_hits, corpus_size))

    # Only save actual data if number of lines doesn't exceed the limit
    if len(lines_list) <= cache_max:
        lines_tuple = tuple(lines_list)
        try:
            mc.add(cache_keys.data_key, lines_tuple)
        except CacheError:
            pass
        return lines_tuple
    return tuple(lines_list)


def count_query_worker(
    ctx: CtxDep,
    corpus: str,
    cqp_query: list[str | list[str]],
    group_by: list[tuple[str, bool]],
    within: str | None,
    ignore_case: Collection[str] = frozenset(),
    expand_prequeries: bool = True,
    use_cache: bool = False,
    cache_max: int = 0,
    abort_signal: AbortSignal | None = None,
) -> tuple[Iterable[str], int, int]:
    """Worker for counting word/attribute occurrences in a corpus.

    Args:
        ctx: The request context.
        corpus: The corpus to query.
        cqp_query: The CQP query or list of queries.
        group_by: List of tuples specifying attributes to group by and whether the attribute is a structural one.
        within: The structural context to limit the search to.
        ignore_case: Set of attributes to ignore case for.
        expand_prequeries: Whether to expand pre-queries to the full 'within' context.
        use_cache: Whether to use caching for the query results.
        cache_max: Maximum number of lines to cache.
        abort_signal: An optional event to signal abortion of the operation.

    Returns:
        A tuple containing:
            - An iterable of result lines from the CQP query.
            - The number of hits in the corpus.
            - The size of the corpus.
    """
    if isinstance(cqp_query[-1], list):
        subcqp = cqp_query[-1]
        base_cqp: list[str] = cast(list[str], cqp_query[:-1])
    else:
        subcqp = None
        base_cqp = cast(list[str], cqp_query)

    cache_keys: _CountCacheKeys | None = None
    if use_cache and ctx.cache:
        cache_keys = _get_count_cache_keys(
            corpus, cqp_query, group_by, within, ignore_case, expand_prequeries, ctx.cache.sync
        )
        zero_hit_result = [cqp.END_OF_LINE] * len(subcqp) if subcqp else []
        cached = _check_count_cache(cache_keys, zero_hit_result, ctx.cache.sync)
        if cached is not None:
            return cached

    optimize = True
    cqpparams = {"within": within}

    cmd = [f"{corpus};"]
    cqpparams_temp = {}
    for i, c in enumerate(base_cqp):
        cqpparams_temp = cqpparams.copy()
        pre_query = i + 1 < len(base_cqp)

        if pre_query and expand_prequeries:
            cqpparams_temp["expand"] = "to " + cast(str, within)

        if optimize:
            cmd += cqp.optimize_query(c, cqpparams_temp, find_match=(not pre_query))[1]
        else:
            cmd += cqp.make_query(cqp.make_cqp(c, **cqpparams_temp))

        if pre_query:
            cmd += ["Last;"]

    cmd += ["size Last;"]
    cmd += ["info; .EOL.;"]

    # TODO: Match targets in a better way
    has_target = any(re.search(r"(?:^|[ ()\]])@(?:\w+:)?\[", x) for x in base_cqp)

    cmd += [
        """tabulate Last {} > "| sort | uniq -c | sort -nr";""".format(
            ", ".join(
                "{} {}{}".format(
                    "target" if has_target else ("match" if g[1] else "match .. matchend"),
                    g[0],
                    " %c" if g[0] in ignore_case else "",
                )
                for g in group_by
            )
        )
    ]

    if subcqp:
        cmd += ["mainresult=Last;"]
        if "expand" in cqpparams_temp:
            del cqpparams_temp["expand"]
        for c in subcqp:
            cmd += [".EOL.;"]
            cmd += ["mainresult;"]
            cmd += cqp.optimize_query(c, cqpparams_temp, find_match=True)[1]
            cmd += [
                """tabulate Last {} > "| sort | uniq -c | sort -nr";""".format(
                    ", ".join(f"match .. matchend {g[0]}" for g in group_by)
                )
            ]

    cmd += ["exit;"]

    lines = ctx.cwb.run_cqp(cmd, abort_signal=abort_signal)

    # Skip CQP version
    next(lines)

    # Size of the query result
    nr_hits = int(next(lines))

    # Get corpus size
    corpus_size = 0
    for line in lines:
        if line.startswith("Size:"):
            _, corpus_size = line.split(":")
            corpus_size = int(corpus_size.strip())
        elif line == cqp.END_OF_LINE:
            break

    if cache_keys is not None:
        lines = _save_count_cache(cache_keys, lines, nr_hits, corpus_size, cache_max, ctx.cache.sync)

    return lines, nr_hits, corpus_size


def count_query_worker_simple(
    ctx: CtxDep,
    corpus: str,
    cqp_query: list[str | list[str]],
    group_by: list[tuple[str, bool]],
    within: str | None = None,
    ignore_case: Collection[str] = frozenset(),
    expand_prequeries: bool = True,
    use_cache: bool = False,
    cache_max: int = 0,
    abort_signal: AbortSignal | None = None,
) -> tuple[Iterable[str], int, int]:
    """Perform a simple count query for all values of the given attributes.

    This is used for simple statistics queries where we can use cwb-scan-corpus. Currently, the CQP query is ignored,
    and all tokens in the corpus are counted.

    Args:
        ctx: The request context.
        corpus: The corpus to query.
        cqp_query: The CQP query or list of queries.
        group_by: List of tuples specifying attributes to group by and whether the attribute is a structural one.
        within: The structural context to limit the search to. Unused in simple count queries.
        ignore_case: Collection of attributes to ignore case for.
        expand_prequeries: Whether to expand pre-queries to the full 'within' context. Unused in simple count queries.
        use_cache: Whether to use caching for the query results.
        cache_max: Maximum number of lines to cache.
        abort_signal: An optional event to signal abortion of the operation.

    Returns:
        A tuple containing:
            - A list of result lines from the CQP query.
            - The number of hits in the corpus.
            - The size of the corpus.
    """
    cache_keys: _CountCacheKeys | None = None
    if use_cache and ctx.cache:
        cache_keys = _get_count_cache_keys(
            corpus, cqp_query, group_by, within, ignore_case, expand_prequeries, ctx.cache.sync
        )
        cached = _check_count_cache(cache_keys, [], ctx.cache.sync)
        if cached is not None:
            return cached

    lines = list(ctx.cwb.run_cwb_scan(corpus, [g[0] for g in group_by], abort_signal=abort_signal))
    nr_hits = 0

    ic_index = []
    new_lines = {}
    if ignore_case:
        ic_index = [i for i, g in enumerate(group_by) if g[0] in ignore_case]

    for i, line in enumerate(lines):
        c, v = line.split("\t", 1)
        nr_hits += int(c)

        if ic_index:
            v = "\t".join(vv.lower() if j in ic_index else vv for j, vv in enumerate(v.split("\t")))
            new_lines[v] = new_lines.get(v, 0) + int(c)
        else:
            # Convert result to the same format as the regular CQP count
            lines[i] = f"{c} {v}"

    if ic_index:
        lines = []
        for v, c in new_lines.items():
            # Convert result to the same format as the regular CQP count
            lines.append(f"{c} {v}")

    if cache_keys is not None:
        lines = _save_count_cache(cache_keys, lines, nr_hits, nr_hits, cache_max, ctx.cache.sync)

    # Corpus size equals number of hits since we count all tokens
    return lines, nr_hits, nr_hits

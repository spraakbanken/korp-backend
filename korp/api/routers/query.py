"""Router for corpus queries."""

import base64
import binascii
import dataclasses
import os
import random
import uuid
import zlib
from collections import defaultdict
from collections.abc import AsyncGenerator, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, cast

import anyio
import anyio.to_thread
from anyio import CapacityLimiter
from fastapi import APIRouter, Query
from pydantic import BeforeValidator
from pydantic.json_schema import SkipJsonSchema

from korp import auth, caching, cqp, utils
from korp.api import params
from korp.config import settings
from korp.cwb import CWB
from korp.dependencies import AbortDep, AbortSignal, CtxDep
from korp.handler import api_handler
from korp.memcached import MemcachedSyncClient

if TYPE_CHECKING:
    import anyio.abc

router = APIRouter(tags=["Concordance"])

LeftContextParam: TypeAlias = Annotated[
    list[str] | SkipJsonSchema[None],
    Query(
        description="The amount of context to show on the left of each match. This overrides 'default_context' for "
        "the specified corpora. Format: `corpus1:context1,corpus2:context2,...`"
    ),
    BeforeValidator(utils.split_csv),
]

RightContextParam: TypeAlias = Annotated[
    list[str] | SkipJsonSchema[None],
    Query(
        description="The amount of context to show on the right of each match. This overrides 'default_context' for "
        "the specified corpora. Format: `corpus1:context1,corpus2:context2,...`"
    ),
    BeforeValidator(utils.split_csv),
]

InOrderParam: TypeAlias = Annotated[bool, Query(description="Whether to perform an in-order search.")]

QueryDataParam: TypeAlias = Annotated[
    str | None, Query(description="Previously saved query data for caching purposes.")
]

RandomSeedParam: TypeAlias = Annotated[int | None, Query(description="Random seed for random sorting.")]

ShowParam: TypeAlias = Annotated[
    Sequence[str],
    Query(description="Comma-separated list of positional attributes to show in the results."),
    BeforeValidator(utils.split_csv),
]

ShowStructParam: TypeAlias = Annotated[
    Sequence[str] | SkipJsonSchema[None],
    Query(description="Comma-separated list of structural attributes to show in the results."),
    BeforeValidator(utils.split_csv),
]

SortParam: TypeAlias = Annotated[
    Literal["keyword", "left", "right", "random"] | str | SkipJsonSchema[None],
    Query(
        description="Sorting method for the results. The sorting is performed *within* each corpus.\n\n"
        "The available options are:\n\n"
        "- `keyword` - Sort by match\n"
        "- `left` - Sort by left context\n"
        "- `right` - Sort by right context\n"
        "- `random` - Random order\n"
        "- Any positional attribute - Sort by given attribute\n\n"
        "It is not possible to sort across corpora, or to sort by structural attributes.\n\n"
        "By default, results are returned in corpus order."
    ),
]

StartParam: TypeAlias = Annotated[
    int,
    Query(description="The index of the first row to return (0-based)."),
]

EndParam: TypeAlias = Annotated[
    int,
    Query(description="The index of the last row to return (0-based, inclusive)."),
]


@dataclass
class QueryParameters:
    """Parameters for query routes, parsed and validated."""

    corpora: list[str]
    cqp_query: list[str]
    start: int = 0
    end: int = 9
    show: set[str] = dataclasses.field(default_factory=lambda: {"word"})
    show_struct: set[str] = dataclasses.field(default_factory=set)
    cut: int | None = None
    sort: str | None = None
    random_seed: int | None = None
    in_order: bool = True
    within: dict[str, str | None] = dataclasses.field(default_factory=dict)
    default_within: str | None = None
    context: defaultdict[str, tuple[str, ...]] = dataclasses.field(default_factory=lambda: defaultdict(tuple))
    default_context: str | None = None
    expand_prequeries: bool = True
    query_data: str | None = None


async def parse_parameters(
    ctx: CtxDep,
    corpus: list[str],
    cqp_query: list[str],
    start: int,
    end: int,
    show: Sequence[str],
    show_struct: Sequence[str] | None,
    cut: int | None = None,
    sort: str | None = None,
    random_seed: int | None = None,
    in_order: bool = True,
    within: Sequence[str] | None = None,
    default_within: str | None = None,
    context: Sequence[str] | None = None,
    default_context: str | None = None,
    left_context: Sequence[str] | None = None,
    right_context: Sequence[str] | None = None,
    expand_prequeries: bool = True,
    query_data: str | None = None,
) -> QueryParameters:
    """Parse and validate query parameters.

    Args:
        ctx: The request context.
        corpus: List of corpora to query.
        cqp_query: List of CQP query strings.
        start: The index of the first row to return (0-based).
        end: The index of the last row to return (0-based, inclusive).
        show: List of positional attributes to show in the results.
        show_struct: List of structural attributes to show in the results.
        cut: Maximum number of results to return per corpus. With this enabled, the total number of results will be
            incorrect.
        sort: Sorting method for the results.
        random_seed: Random seed for random sorting.
        in_order: Whether to perform an in-order search.
        within: List of "within" specifications for each corpus.
        default_within: Default "within" specification if not provided for a corpus.
        context: List of context specifications for each corpus.
        default_context: Default context specification if not provided for a corpus.
        left_context: List of left context specifications for each corpus.
        right_context: List of right context specifications for each corpus.
        expand_prequeries: Whether to expand prequeries when multiple CQP queries are provided.
        query_data: Previously saved query data for caching purposes.

    Returns:
        A QueryParameters object containing the parsed and validated parameters.

    Raises:
        ValueError: If any of the parameters are invalid.
    """
    corpora = corpus or []
    await auth.check_authorization(corpora, ctx)

    show_set = set(show)
    show_set.add("word")  # Always include word

    show_structs = set(show_struct) if show_struct else set()

    if settings.MAX_KWIC_ROWS and end - start >= settings.MAX_KWIC_ROWS:
        raise ValueError(f"At most {settings.MAX_KWIC_ROWS} KWIC rows can be returned per call.")

    within_dict = cqp.parse_within(within, default_within)

    # Parse context/left_context/right_context/default_context
    context_dict: defaultdict[str, tuple[str, ...]] = defaultdict(lambda: (default_context or "",))
    contexts = {}

    for context_type, context_pairs in {
        "left_context": left_context,
        "right_context": right_context,
        "context": context,
    }.items():
        if context_pairs:
            for pair in context_pairs:
                if ":" not in pair:
                    raise ValueError(f"Malformed value for key '{context_type}'.")
                contexts[context_type] = {
                    context_corpus.upper(): value
                    for context_corpus, value in (pair.split(":", 1) for pair in context_pairs)
                }
        else:
            contexts[context_type] = {}

    for context_corpus in {c for v in contexts.values() for c in v}:
        if context_corpus in contexts["left_context"] or context_corpus in contexts["right_context"]:
            context_dict[context_corpus] = (
                contexts["left_context"].get(context_corpus, default_context),
                contexts["right_context"].get(context_corpus, default_context),
            )
        else:
            context_dict[context_corpus] = (contexts["context"].get(context_corpus, default_context),)

    cqp_query = [c.strip().removesuffix(";") for c in cqp_query]

    if len(cqp_query) > 1 and expand_prequeries and not all(within_dict[c] for c in corpora):
        raise ValueError("Multiple CQP queries requires 'within' or 'expand_prequeries=false'")

    return QueryParameters(
        corpora=corpora,
        cqp_query=cqp_query,
        within=within_dict,
        start=start,
        end=end,
        show=show_set,
        show_struct=show_structs,
        cut=cut,
        sort=sort,
        random_seed=random_seed,
        in_order=in_order,
        default_within=default_within,
        default_context=default_context,
        context=context_dict,
        expand_prequeries=expand_prequeries,
        query_data=query_data,
    )


async def perform_query(
    query_params: QueryParameters, ctx: CtxDep, abort_signal: AbortSignal | None = None
) -> AsyncGenerator[dict]:
    """Execute a corpus query and stream KWIC results.

    Args:
        query_params: Parsed and validated query parameters.
        ctx: Request-scoped context containing common parameters and services.
        abort_signal: Optional abort handle that can be used to cancel a running query.

    Yields:
        Dictionaries containing KWIC rows and related metadata for the requested query.
    """
    incremental = ctx.common.incremental
    use_cache = ctx.common.cache
    free_search = not query_params.in_order

    corpora = query_params.corpora
    cqp_query = query_params.cqp_query
    within = query_params.within
    cut = query_params.cut
    expand_prequeries = query_params.expand_prequeries
    query_data = query_params.query_data
    start = query_params.start
    end = query_params.end

    result: dict[str, Any] = {"kwic": []}

    # Checksum for whole query, used to verify query_data from the client
    checksum = utils.get_hash((sorted(corpora), cqp_query, sorted(within.items()), cut, expand_prequeries, free_search))

    debug = {}
    if ctx.common.debug:
        debug["checksum"] = checksum

    total_hits = 0
    corpus_hit_stats = {}

    cached_corpus_hit_stats = {}  # Information about which corpora have how many hits, either from query_data or cache

    # The query_data parameter contains previously saved info about corpus hit counts (cached_corpus_hit_stats)
    if query_data:
        try:
            query_data = zlib.decompress(
                base64.b64decode(query_data.replace("\\n", "\n").replace("-", "+").replace("_", "/"))
            ).decode("UTF-8")
        except Exception:
            if ctx.common.debug:
                debug["query_data_unparseable"] = True
        else:
            if ctx.common.debug:
                debug["query_data_read"] = True
            saved_checksum, stats_temp = query_data.split(";", 1)
            if saved_checksum == checksum:
                for pair in stats_temp.split(";"):
                    corpus, hits = pair.split(":")
                    cached_corpus_hit_stats[corpus] = int(hits)
            elif ctx.common.debug:
                debug["query_data_checksum_mismatch"] = True

    # If we have no usable query_data, try to get cached corpus hit counts from memcached instead
    if use_cache and not cached_corpus_hit_stats:
        memcached_keys = {}
        cache_prefixes = await caching.cache_prefix(ctx.cache, [corpus.split("|")[0] for corpus in corpora])
        for corpus in corpora:
            corpus_checksum = utils.get_hash((cqp_query, within[corpus], cut, expand_prequeries, free_search))
            memcached_keys[f"{cache_prefixes[corpus.split('|')[0]]}:query_size_{corpus_checksum}"] = corpus

        cached_corpus_hits = await ctx.cache.get_many(memcached_keys.keys())
        for key in cached_corpus_hits:
            cached_corpus_hit_stats[memcached_keys[key]] = cached_corpus_hits[key]

    start_local = start
    end_local = end

    if cached_corpus_hit_stats:
        if ctx.common.debug:
            debug["cache_coverage"] = f"{len(cached_corpus_hit_stats)}/{len(corpora)}"
        complete_hits = set(corpora) == set(cached_corpus_hit_stats.keys())
    else:
        complete_hits = False

    if abort_signal and abort_signal.is_set():
        return

    if complete_hits:
        # We have cached_corpus_hit_stats available for all corpora, so calculate which corpora need to be queried and
        # then query them in parallel
        corpora_hits = which_hits(corpora, cached_corpus_hit_stats, start, end)
        total_hits = sum(cached_corpus_hit_stats.values())
        corpus_hit_stats = cached_corpus_hit_stats
        corpora_kwics = {}
        progress_count = 0

        if len(corpora_hits) > 0:
            if incremental:
                yield {"progress_corpora": list(corpora_hits.keys())}

            limiter = CapacityLimiter(settings.PARALLEL_THREADS)
            send, receive = anyio.create_memory_object_stream(0)

            async def _worker(corpus: str, send_channel: anyio.abc.ObjectSendStream) -> None:
                async with send_channel:
                    if abort_signal and abort_signal.is_set():
                        return
                    try:
                        kwic, _ = await anyio.to_thread.run_sync(
                            partial(
                                query_and_parse,
                                query_params,
                                corpus,
                                start=corpora_hits[corpus][0],
                                end=corpora_hits[corpus][1],
                                cwb=ctx.cwb,
                                mc=ctx.cache.sync,
                                use_cache=use_cache,
                                abort_signal=abort_signal,
                            ),
                            limiter=limiter,
                        )
                    except Exception as e:
                        raise cqp.CQPError(e) from e

                    await send_channel.send((corpus, kwic))

            async with anyio.create_task_group() as tg:
                for corpus in corpora_hits:
                    tg.start_soon(_worker, corpus, send.clone())

                await send.aclose()

                async for corpus, kwic in receive:
                    if abort_signal and abort_signal.is_set():
                        tg.cancel_scope.cancel()
                        return

                    corpora_kwics[corpus] = kwic
                    if incremental:
                        yield {
                            f"progress_{progress_count}": {
                                "corpus": corpus,
                                "hits": corpora_hits[corpus][1] - corpora_hits[corpus][0] + 1,
                            }
                        }
                        progress_count += 1

            for corpus in corpora:
                if corpus in corpora_hits:
                    result["kwic"].extend(corpora_kwics[corpus])
    else:
        # cached_corpus_hit_stats is missing or incomplete, so we need to query the corpora in
        # serial until we have the needed rows, and then query the remaining corpora
        # in parallel to get number of hits.
        if incremental:
            yield {"progress_corpora": corpora}
        progress_count = 0
        rest_corpora: list[str] = []

        # Serial until we've got all the requested rows
        for i, corpus in enumerate(corpora):
            if abort_signal and abort_signal.is_set():
                return
            if end_local < 0:
                rest_corpora = corpora[i:]
                break
            skip_corpus = False
            if corpus in cached_corpus_hit_stats:
                nr_hits = cached_corpus_hit_stats[corpus]
                if nr_hits - 1 < start_local:
                    kwic = []
                    skip_corpus = True

            if not skip_corpus:
                kwic, nr_hits = await anyio.to_thread.run_sync(
                    partial(
                        query_and_parse,
                        query_params,
                        corpus,
                        start=start_local,
                        end=end_local,
                        cwb=ctx.cwb,
                        mc=ctx.cache.sync,
                        use_cache=use_cache,
                        abort_signal=abort_signal,
                    )
                )

            corpus_hit_stats[corpus] = nr_hits
            total_hits += nr_hits

            # Calculate which hits from next corpus we need, if any
            start_local -= nr_hits
            end_local -= nr_hits
            start_local = max(start_local, 0)

            result["kwic"].extend(kwic)

            if incremental:
                yield {f"progress_{progress_count}": {"corpus": corpus, "hits": nr_hits}}
                progress_count += 1

        if incremental:
            yield result
            result = {}

        if rest_corpora:
            if cached_corpus_hit_stats:
                for corpus in rest_corpora:
                    if corpus in cached_corpus_hit_stats:
                        corpus_hit_stats[corpus] = cached_corpus_hit_stats[corpus]
                        total_hits += cached_corpus_hit_stats[corpus]

            limiter = CapacityLimiter(settings.PARALLEL_THREADS)
            send, receive = anyio.create_memory_object_stream(0)

            async def _worker(corpus: str, send_channel: anyio.abc.ObjectSendStream) -> None:
                async with send_channel:
                    if abort_signal and abort_signal.is_set():
                        return
                    try:
                        _, nr_hits, _ = await anyio.to_thread.run_sync(
                            partial(
                                query_corpus,
                                query_params,
                                corpus,
                                start=0,
                                end=0,
                                cwb=ctx.cwb,
                                mc=ctx.cache.sync,
                                no_results=True,
                                use_cache=use_cache,
                                abort_signal=abort_signal,
                            ),
                            limiter=limiter,
                        )
                    except Exception as e:
                        raise cqp.CQPError(e) from e

                    await send_channel.send((corpus, nr_hits))

            async with anyio.create_task_group() as tg:
                for corpus in rest_corpora:
                    if corpus not in cached_corpus_hit_stats:
                        tg.start_soon(_worker, corpus, send.clone())

                await send.aclose()

                async for corpus, nr_hits in receive:
                    if abort_signal and abort_signal.is_set():
                        tg.cancel_scope.cancel()
                        return
                    corpus_hit_stats[corpus] = nr_hits
                    total_hits += nr_hits
                    if incremental:
                        yield {f"progress_{progress_count}": {"corpus": corpus, "hits": nr_hits}}
                        progress_count += 1

    if ctx.common.debug:
        debug["cqp"] = cqp_query

    result["hits"] = total_hits
    result["corpus_hits"] = corpus_hit_stats
    result["corpus_order"] = corpora
    result["query_data"] = (
        binascii.b2a_base64(
            zlib.compress(bytes(checksum + ";" + ";".join(f"{c}:{h}" for c, h in corpus_hit_stats.items()), "utf-8"))
        )
        .decode("utf-8")
        .replace("+", "-")
        .replace("/", "_")
    )

    if debug:
        result["DEBUG"] = debug

    yield result


@router.get("/query_sample", response_model=dict)
@router.post("/query_sample", response_model=dict, include_in_schema=False)
@api_handler
async def query_sample(
    ctx: CtxDep,
    corpus: params.CorpusParam,
    cqp_query: params.CQPParam,
    show: ShowParam = ("word",),
    show_struct: ShowStructParam = (),
    random_seed: RandomSeedParam = None,
    in_order: InOrderParam = True,
    within: params.WithinParam = None,
    default_within: params.DefaultWithinParam | None = None,
    context: params.ContextParam | None = None,
    default_context: params.DefaultContextParam = "10 words",
    left_context: LeftContextParam | None = None,
    right_context: RightContextParam | None = None,
    expand_prequeries: params.ExpandPrequeriesParam = True,
    query_data: QueryDataParam = None,
    abort_signal: AbortDep = None,
) -> AsyncGenerator[dict]:
    """Perform a CQP query and return a random match.

    The query is performed sequentially on the selected corpora in random order until a match is found. No total hit
    count is calculated.

    Yields:
        A single KWIC result as a dictionary, or an empty result if no matches are found.
    """
    query_params = await parse_parameters(
        ctx=ctx,
        corpus=corpus,
        cqp_query=cqp_query,
        start=0,
        end=0,
        show=show,
        show_struct=show_struct,
        cut=1,
        sort="random",
        random_seed=random_seed,
        in_order=in_order,
        within=within,
        default_within=default_within,
        default_context=default_context,
        left_context=left_context,
        right_context=right_context,
        context=context,
        expand_prequeries=expand_prequeries,
        query_data=query_data,
    )

    corpora = query_params.corpora
    random.shuffle(corpora)

    for c in corpora:
        params_corpus = dataclasses.replace(query_params, corpora=[c])
        async for item in perform_query(params_corpus, ctx, abort_signal=abort_signal):
            if item.get("hits", 0) > 0:
                yield item
                return


@router.get("/query", response_model=None)
@router.post("/query", response_model=None, include_in_schema=False)
@api_handler
async def query(
    ctx: CtxDep,
    corpus: params.CorpusParam,
    cqp_query: params.CQPParam,
    start: StartParam = 0,
    end: EndParam = 9,
    show: ShowParam = ("word",),
    show_struct: ShowStructParam = None,
    cut: int | None = None,
    sort: SortParam = None,
    random_seed: RandomSeedParam = None,
    in_order: InOrderParam = True,
    within: params.WithinParam = None,
    default_within: params.DefaultWithinParam = None,
    default_context: params.DefaultContextParam = "10 words",
    left_context: LeftContextParam = None,
    right_context: RightContextParam = None,
    context: params.ContextParam = None,
    expand_prequeries: params.ExpandPrequeriesParam = True,
    query_data: QueryDataParam = None,
    abort_signal: AbortDep = None,
) -> AsyncGenerator[dict]:
    """Perform a CQP query and return a number of matches.

    Yields:
        KWIC results as dictionaries, followed by a final dictionary containing the total hit count and other metadata.
    """
    query_params = await parse_parameters(
        ctx=ctx,
        corpus=corpus,
        cqp_query=cqp_query,
        start=start,
        end=end,
        show=show,
        show_struct=show_struct,
        cut=cut,
        sort=sort,
        random_seed=random_seed,
        in_order=in_order,
        within=within,
        default_within=default_within,
        default_context=default_context,
        left_context=left_context,
        right_context=right_context,
        context=context,
        expand_prequeries=expand_prequeries,
        query_data=query_data,
    )

    async for item in perform_query(query_params, ctx, abort_signal=abort_signal):
        yield item


def query_corpus(
    query_params: QueryParameters,
    corpus: str,
    start: int,
    end: int,
    cwb: CWB,
    mc: MemcachedSyncClient,
    no_results: bool = False,
    use_cache: bool = False,
    abort_signal: AbortSignal | None = None,
) -> tuple[Iterable[str], int, dict]:
    """Perform a CQP query on a single corpus and return parsed results.

    Args:
        query_params: Parsed and validated query parameters.
        corpus: Corpus to query.
        start: Start index of results to return.
        end: End index of results to return.
        cwb: CWB instance to use for querying.
        no_results: If True, do not return any KWIC rows, only the number of hits.
        use_cache: Whether to use caching for this query.
        mc: Memcached client to use for caching, if use_cache is True.
        abort_signal: Optional abort handle that can be used to cancel a running query.

    Returns:
        A tuple containing:
            - An iterable of KWIC rows (empty if no_results is True).
            - The total number of hits for the query.
            - A dictionary with additional metadata (currently unused).

    Raises:
        CQPError: If the CQP query fails.
    """
    cqp_query = query_params.cqp_query
    show = query_params.show
    within = query_params.within[corpus]
    context = query_params.context[corpus]
    show_structs = query_params.show_struct
    expand_prequeries = query_params.expand_prequeries
    free_search = not query_params.in_order
    cut = query_params.cut
    sort = query_params.sort
    random_seed = query_params.random_seed

    cache_dir = settings.CACHE_DIR
    cache_max_query_data = settings.CACHE_MAX_QUERY_DATA

    if use_cache and cache_dir:
        # Calculate checksum (needs to contain all arguments that may influence the results)
        checksum_data = (cqp_query, within, cut, expand_prequeries, free_search)

        checksum = utils.get_hash(checksum_data)
        unique_id = str(uuid.uuid4())

        cache_query = f"query_data_{checksum}"
        cache_query_temp = cache_query + "_" + unique_id

        corpus_base = corpus.split("|", 1)[0]
        cache_filename = Path(cache_dir) / f"{corpus_base}:query_data_{checksum}"
        cache_filename_temp = cache_filename.with_name(cache_filename.name + "_" + unique_id)

        cache_size_key = f"{caching.cache_prefix_sync(mc, corpus_base)}:query_size_{checksum}"
        cache_hits = mc.get(cache_size_key)
        is_cached = cache_hits is not None and cache_filename.is_file()
        cached_no_hits = cache_hits == 0
    else:
        is_cached = False

    # CQP optimization is currently always enabled
    optimize = True

    show = show.copy()  # To not edit the original

    cqpparams = {"within": within, "cut": cut}

    # Handle aligned corpora
    if "|" in corpus:
        linked = corpus.split("|")
        cqp_final = []

        for c in cqp_query:
            cs = c.split("LINKED_CORPUS:")

            # In a multi-language query, the "within" argument must be placed directly
            # after the main (first language) query
            if len(cs) > 1 and within:
                cs[0] = f"{cs[0].rstrip()[:-1]} within {within} : "
                del cqpparams["within"]

            cc = [cs[0]]

            for d in cs[1:]:
                linked_corpora, link_cqp = d.split(None, 1)
                if linked[1] in linked_corpora.split("|"):
                    cc.append(f"{linked[1]} {link_cqp}")

            cqp_final.append("".join(cc).rstrip(": "))

        cqp_query = cqp_final
        corpus = linked[0]
        show.add(linked[1].lower())

    # Sorting
    if sort == "left":
        sortcmd = ["sort by word on match[-1] .. match[-3];"]
    elif sort == "keyword":
        sortcmd = ["sort by word;"]
    elif sort == "right":
        sortcmd = ["sort by word on matchend[1] .. matchend[3];"]
    elif sort == "random":
        sortcmd = [f"sort randomize {random_seed or ''};"]
    elif sort:
        # Sort by positional attribute
        sortcmd = [f"sort by {sort};"]
    else:
        sortcmd = []

    # Build the CQP query
    cmd = []

    if use_cache:
        cmd.append(f'set DataDirectory "{cache_dir}";')

    cmd.append(f"{corpus};")

    # This prints the attributes and their relative order:
    cmd += cwb.show_attributes()

    retcode = cqp.QueryOptimizeResult.SUCCESS

    if is_cached:
        # This exact query has been done before. Read corpus positions from cache.
        if not cached_no_hits:
            cmd.append(f"Last = {cache_query};")
            # Touch cache file to delay its removal
            os.utime(cache_filename)
    else:
        for i, c in enumerate(cqp_query):
            cqpparams_temp = cqpparams.copy()
            pre_query = i + 1 < len(cqp_query)

            if pre_query and expand_prequeries:
                cqpparams_temp["expand"] = "to " + cast(str, within)

            if free_search:
                retcode, free_query = cqp.optimize_query(c, cqpparams_temp, free_search=True)
                if retcode == cqp.QueryOptimizeResult.NOT_POSSIBLE:
                    raise cqp.CQPError("Couldn't convert into free order query.")
                cmd += free_query
            elif optimize and expand_prequeries:
                # We can only optimize when expand_prequeries is enabled
                cmd += cqp.optimize_query(c, cqpparams_temp, find_match=(not pre_query))[1]
            else:
                cmd += cqp.make_query(cqp.make_cqp(c, **cqpparams_temp))

            if pre_query:
                cmd.append("Last;")

    if use_cache and cached_no_hits:
        # Print EOL if no hits
        cmd.append(".EOL.;")
    else:
        # This prints the size of the query (i.e., the number of results):
        cmd.append("size Last;")

    if use_cache and not is_cached:
        cmd.append(f"{cache_query_temp} = Last; save {cache_query_temp};")

    if not no_results and not (use_cache and cached_no_hits):
        if free_search and retcode == cqp.QueryOptimizeResult.SUCCESS:
            tokens, _ = cqp.parse_cqp(cqp_query[-1])
            cmd.append("Last;")
            cmd.append(f"cut {start} {end};")
            cmd += cqp.make_query(cqp.make_cqp(f"({' | '.join(set(tokens))})", **cqpparams))

        cmd.append(f"show +{' +'.join(show)};")
        if len(context) == 1:
            cmd.append(f"set Context {context[0]};")
        else:
            cmd.append(f"set LeftContext {context[0]};")
            cmd.append(f"set RightContext {context[1]};")
        cmd.append(f"set LeftKWICDelim '{cqp.LEFT_DELIM} '; set RightKWICDelim ' {cqp.RIGHT_DELIM}';")
        if show_structs:
            cmd.append(f"set PrintStructures '{', '.join(show_structs)}';")
        cmd.append("set ExternalSort yes;")
        cmd += sortcmd
        if free_search and retcode == cqp.QueryOptimizeResult.SUCCESS:
            # The results are already cut to the right range, so print all of them
            cmd.append("cat Last;")
        else:
            cmd.append(f"cat Last {start} {end};")

    cmd.append("exit;")

    # Then we call the CQP binary, and read the results
    lines = cwb.run_cqp(cmd, attr_ignore=True, abort_signal=abort_signal)

    # Skip the CQP version
    next(lines)

    # Remove cache file if it exceeds max cache file size
    if use_cache and not is_cached and cache_max_query_data:
        cache_file = Path(cache_dir) / f"{corpus}:{cache_query_temp}"
        try:
            if cache_file.is_file() and cache_file.stat().st_size > cache_max_query_data:
                cache_file.unlink()
        except FileNotFoundError:
            pass

    # Read the attributes and their relative order
    attrs = cwb.read_attributes(lines)

    # Read the size of the query, i.e., the number of results
    nr_hits = next(lines)
    nr_hits = 0 if nr_hits == cqp.END_OF_LINE else int(nr_hits)

    if use_cache and not is_cached and not cached_no_hits:
        # Save number of hits
        mc.add(cache_size_key, nr_hits)

        try:
            cache_filename_temp.rename(cache_filename)
        except FileNotFoundError:
            pass

    return lines, nr_hits, attrs


def _parse_line_header(line: str) -> tuple[str | None, int | None, str]:
    """Parse the header portion of a concordance line.

    Args:
        line: Raw concordance line from CWB.

    Returns:
        A tuple of (aligned_corpus, position, remaining_line).
        If this is an aligned corpus result, aligned_corpus is set and position is None.
        Otherwise, aligned_corpus is None and position contains the match position.
    """
    header, remainder = line.split(":", 1)
    if header.startswith("-->"):
        # For aligned corpora, every other line is the aligned result
        return header[3:], None, remainder
    # This is the result row for the query corpus
    return None, int(header), remainder


def _parse_line_structs(line: str, ls_attrs: set[str]) -> tuple[dict[str, str | None], str]:
    """Parse PrintStructures from a concordance line.

    Args:
        line: The line content after the header.
        ls_attrs: Set of structural attributes to extract.

    Returns:
        A tuple of (linestructs dict, remaining line content).
    """
    if not ls_attrs:
        return {}, line

    if ":  " in line:
        lineattr, line = line.rsplit(":  ", 1)
    else:
        # Sometimes, depending on context, CWB uses only one space instead of two as a separator
        lineattr, line = line.split(">: ", 1)
        lineattr += ">"

    lineattrs = lineattr[2:-1].split("><")

    # Handle "><" in attribute values
    if len(lineattrs) != len(ls_attrs):
        new_lineattrs = []
        for la in lineattrs:
            if la.split(" ", 1)[0] not in ls_attrs:
                new_lineattrs[-1] += "><" + la
            else:
                new_lineattrs.append(la)
        lineattrs = new_lineattrs

    linestructs: dict[str, str | None] = {}
    for s in lineattrs:
        if s in ls_attrs:
            linestructs[s] = None
        else:
            s_key, s_val = s.split(" ", 1)
            linestructs[s_key] = s_val

    return linestructs, line


@dataclass
class _TokenParseState:
    """State maintained while parsing tokens from a concordance line."""

    struct: str | None = None
    struct_value: list[str] = dataclasses.field(default_factory=list)
    structs: dict = dataclasses.field(default_factory=dict)
    token_index: int = 0


def _parse_tokens(
    words: list[str],
    p_attrs: list[str],
    s_attrs: set[str],
) -> tuple[list[dict], dict[str, int]]:
    """Parse tokens from concordance line words.

    Args:
        words: List of space-separated words from the concordance line.
        p_attrs: List of positional attributes to extract.
        s_attrs: Set of structural attributes to recognize.

    Returns:
        A tuple of (tokens list, match dict with 'start' and 'end' keys).
    """
    nr_splits = len(p_attrs) - 1
    state = _TokenParseState()
    tokens: list[dict] = []
    match: dict[str, int] = {}

    for raw_word in words:
        word = raw_word
        if state.struct:
            # Structural attrs can be split in the middle (<s_n 123>),
            # so we need to finish the structure here
            if ">" not in word:
                state.struct_value.append(word)
                continue

            struct_v, word = word.split(">", 1)
            struct_tag, struct_attr = state.struct.split("_", 1)
            state.structs.setdefault("open", {}).setdefault(struct_tag, {})
            state.structs["open"][struct_tag][struct_attr] = " ".join([*state.struct_value, struct_v])
            state.struct = None
            state.struct_value = []

        # We use special delimiters to see when we enter and leave the match region
        if word == cqp.LEFT_DELIM:
            match["start"] = state.token_index
            continue
        if word == cqp.RIGHT_DELIM:
            match["end"] = state.token_index
            continue

        # We read all structural attributes that are opening (from the left)
        while word[0] == "<":
            if word[1:] in s_attrs:
                # We have found a structural attribute with a value (<s_n 123>).
                # We continue to the next word to get the value
                state.struct = word[1:]
                break
            if ">" in word and word[1 : word.find(">")] in s_attrs:
                # We have found a structural attribute without a value (<s>)
                struct_name, word = word[1:].split(">", 1)
                state.structs.setdefault("open", {}).setdefault(struct_name, {})
            else:
                # What we've found is not a structural attribute
                break

        if state.struct:
            # If we stopped in the middle of a struct (<s_n 123>),
            # we need to continue with the next word
            continue

        # Now we read all s-attrs that are closing (from the right)
        while word[-1] == ">" and "</" in word:
            tempword, closing_struct = word[:-1].rsplit("</", 1)
            if not tempword or closing_struct not in s_attrs:
                break
            word = tempword
            state.structs.setdefault("close", [])
            closing_tag = closing_struct.split("_")[0]
            if closing_tag not in state.structs["close"]:
                state.structs["close"].insert(0, closing_tag)

        # What's left is the word with its p-attrs
        values = word.rsplit("/", nr_splits)
        token: dict[str, str | dict | None] = {
            attr: cqp.translate_undef(val) for (attr, val) in zip(p_attrs, values, strict=True)
        }
        if state.structs:
            # Convert dict into list
            if "open" in state.structs:
                state.structs["open"] = [{k: v} for k, v in state.structs["open"].items()]
            token["structs"] = state.structs
            state.structs = {}
        tokens.append(token)
        state.token_index += 1

    return tokens, match


def query_parse_lines(
    query_params: QueryParameters,
    corpus: str,
    lines: Iterable[str],
    attrs: dict[str, list[str]],
    abort_signal: AbortSignal | None = None,
) -> list[dict]:
    """Parse concordance lines from CWB.

    Args:
        query_params: Parsed query parameters.
        corpus: Name of the corpus being queried.
        lines: Iterable of raw concordance lines from CWB.
        attrs: Dictionary of available attributes by type.
        abort_signal: Optional signal to abort processing.

    Returns:
        List of KWIC row dictionaries.
    """
    show = query_params.show
    show_structs = query_params.show_struct
    free_search = not query_params.in_order

    # Filter out unavailable attributes
    p_attrs = [attr for attr in attrs["p"] if attr in show]
    s_attrs = {attr for attr in attrs["s"] if attr in show}
    ls_attrs = {attr for attr in attrs["s"] if attr in show_structs}

    last_line_span: tuple[int, int] | tuple[()] = ()
    kwic: list[dict] = []

    for raw_line in lines:
        if abort_signal and abort_signal.is_set():
            return []

        # Parse header to get aligned corpus info and position
        aligned, position, line = _parse_line_header(raw_line)

        # Parse PrintStructures (only for non-aligned lines)
        if aligned:
            linestructs = {}
        else:
            linestructs, line = _parse_line_structs(line, ls_attrs)

        # Parse tokens
        words = line.split()
        try:
            tokens, match = _parse_tokens(words, p_attrs, s_attrs)
        except (IndexError, ValueError):
            # Attributes containing ">" or "<" can make some lines unparseable. We skip them
            # until we come up with a better solution.
            continue

        if position is not None:
            match["position"] = position

        # Handle aligned corpus results
        if aligned:
            if words != ["(no", "alignment", "found)"]:
                kwic[-1].setdefault("aligned", {})[aligned] = tokens
            continue

        # Skip rows where match start wasn't found (CQP bug with long sentences)
        if "start" not in match:
            continue

        # Build KWIC row
        kwic_row: dict[str, Any] = {
            "corpus": corpus,
            "match": match if not free_search else [match],
        }
        if linestructs:
            kwic_row["structs"] = linestructs
        kwic_row["tokens"] = tokens

        # Handle free search deduplication
        if free_search:
            line_span = (match["position"] - match["start"], match["position"] - match["start"] + len(tokens) - 1)
            if line_span == last_line_span:
                kwic[-1]["match"].append(match)
            else:
                kwic.append(kwic_row)
            last_line_span = line_span
        else:
            kwic.append(kwic_row)

    return kwic


def query_and_parse(
    query_params: QueryParameters,
    corpus: str,
    start: int,
    end: int,
    cwb: CWB,
    mc: MemcachedSyncClient,
    no_results: bool = False,
    use_cache: bool = False,
    abort_signal: AbortSignal | None = None,
) -> tuple[list[dict], int]:
    """Perform a CQP query on a single corpus and return parsed results.

    Args:
        query_params: Parsed and validated query parameters.
        corpus: Corpus to query.
        start: Start index of results to return.
        end: End index of results to return.
        cwb: CWB instance to use for querying.
        mc: Memcached client to use for caching, if use_cache is True.
        no_results: If True, do not return any KWIC rows, only the number of hits.
        use_cache: Whether to use caching for this query.
        abort_signal: Optional abort handle that can be used to cancel a running query.

    Returns:
        A tuple containing:
            - A list of KWIC rows (empty if no_results is True).
            - The total number of hits for the query.
    """
    lines, nr_hits, attrs = query_corpus(
        query_params,
        corpus,
        start=start,
        end=end,
        cwb=cwb,
        mc=mc,
        no_results=no_results,
        use_cache=use_cache,
        abort_signal=abort_signal,
    )
    kwic = query_parse_lines(query_params, corpus, lines, attrs, abort_signal=abort_signal)
    return kwic, nr_hits


def which_hits(corpora: list, stats: dict, start: int, end: int) -> dict[str, tuple[int, int]]:
    """Given total hit counts for each corpus, determine which corpora contain hits in the requested range.

    Args:
        corpora: List of corpus names.
        stats: Dict mapping corpus names to total hit counts.
        start: Global start index.
        end: Global end index.

    Returns:
        Dict mapping corpus names to (start, end) hit ranges within that corpus.
    """
    corpus_hits = {}
    for corpus in corpora:
        hits = stats[corpus]
        if hits > start:
            corpus_hits[corpus] = (start, min(hits - 1, end))

        start -= hits
        end -= hits
        start = max(start, 0)
        if end < 0:
            break

    return corpus_hits

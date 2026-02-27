"""Attribute values route."""

import itertools
from collections import defaultdict
from collections.abc import AsyncIterator
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Annotated

from pydantic import BeforeValidator

from korp.api import params

if TYPE_CHECKING:
    import anyio.abc
import anyio
from anyio import CapacityLimiter
from fastapi import APIRouter, Query

from korp import utils
from korp.config import settings
from korp.memcached import CacheError

from . import count as count_route

router = APIRouter()


@router.get("/attr_values", response_model=dict)
@router.post("/attr_values", response_model=dict, include_in_schema=False)
@utils.api_handler
async def attr_values(
    ctx: utils.CtxDep,
    corpus: params.CorpusParam,
    attr: Annotated[
        list[str], Query(description="Comma-separated list of structural attributes."), BeforeValidator(utils.split_csv)
    ],
    count: Annotated[bool, Query(description="Whether to include counts for each attribute value.")] = False,
    per_corpus: params.PerCorpusParam = True,
    combined: params.CombinedParam = True,
    split: params.SplitParam = None,
) -> AsyncIterator[dict]:
    """Get all available values for one or more structural attributes.

    Args:
        ctx: Request context.
        corpus: Comma-separated list of corpora.
        attr: Comma-separated list of structural attributes.
        count: Whether to include counts for each attribute value.
        per_corpus: Whether to include per-corpus results.
        combined: Whether to include combined results across corpora.
        split: Comma-separated list of attributes to split values for.

    Yields:
        Attribute values (and counts, if requested) for the specified corpora and attributes.
    """
    incremental = ctx.common.incremental
    include_count = count

    await utils.check_authorization(corpus, ctx)

    split = split or []
    split_set = set(split)
    result = {"corpora": defaultdict(dict), "combined": {}}
    from_cache = set()  # Keep track of what has been read from cache
    cache_prefixes: dict[str, str] = {}  # Reused between cache read and write phases

    if ctx.common.cache:
        all_cache = True
        for c in corpus:
            cache_prefixes[c] = await utils.cache_prefix(ctx.cache, c)
            for attribute in attr:
                checksum = utils.get_hash((c, attribute, split, include_count))
                data = await ctx.cache.get(f"{cache_prefixes[c]}:attr_values_{checksum}")
                if data is not None:
                    result["corpora"][c][attribute] = data
                    if ctx.common.debug:
                        result.setdefault("DEBUG", {"caches_read": []})
                        result["DEBUG"]["caches_read"].append(f"{c}:{attribute}")
                    from_cache.add((c, attribute))
                else:
                    all_cache = False
    else:
        all_cache = False

    if not all_cache:
        progress_count = 0
        if incremental:
            yield {"progress_corpora": list(corpus)}

        limiter = CapacityLimiter(settings.PARALLEL_THREADS)
        send, receive = anyio.create_memory_object_stream(0)

        async def _worker(corpus: str, attr: str, send_channel: anyio.abc.ObjectSendStream) -> None:
            """Worker function to run count query in thread."""
            async with send_channel:  # Closes the channel when done
                lines, _nr_hits, _corpus_size = await anyio.to_thread.run_sync(  # type: ignore
                    partial(  # Use partial to be able to pass keyword arguments
                        count_route.count_query_worker_simple,
                        ctx=ctx,
                        corpus=corpus,
                        cqp=[],
                        group_by=[(s, True) for s in attr.split(">")],
                        use_cache=ctx.common.cache,
                    ),
                    limiter=limiter,
                )
                await send_channel.send((corpus, attr, lines))

        async with anyio.create_task_group() as tg:
            for c in corpus:
                for attribute in attr:
                    if (c, attribute) not in from_cache:
                        tg.start_soon(_worker, c, attribute, send.clone())

            await send.aclose()  # Close the original send channel

            async for c, attribute, lines in receive:
                corpus_stats_dict: dict[str, int] = {}
                corpus_stats_set: set[str] = set()
                vals_dict: dict = {}
                attr_parts = attribute.split(">")
                is_nested = len(attr_parts) > 1

                for line in lines:
                    freq_str, raw_val = line.lstrip().split(" ", 1)
                    freq = int(freq_str)

                    if is_nested:
                        vals = raw_val.split("\t")

                        if split_set:
                            vals = [
                                [x for x in v.split("|") if x] if attr_parts[i] in split_set and v else [v]
                                for i, v in enumerate(vals)
                            ]
                            vals_prod = itertools.product(*vals)
                        else:
                            vals_prod = [vals]

                        for combo in vals_prod:
                            if include_count:
                                cur = vals_dict
                                for part in combo[:-1]:
                                    cur = cur.setdefault(part, {})
                                cur[combo[-1]] = cur.get(combo[-1], 0) + freq
                            else:
                                cur = vals_dict
                                for part in combo[:-2]:
                                    cur = cur.setdefault(part, {})
                                cur.setdefault(combo[-2], []).append(combo[-1])
                    else:
                        split_vals = (
                            ([x for x in raw_val.split("|") if x] if raw_val else [""])
                            if attribute in split_set
                            else [raw_val]
                        )
                        for v in split_vals:
                            if include_count:
                                corpus_stats_dict[v] = freq
                            else:
                                corpus_stats_set.add(v)

                if is_nested:
                    result["corpora"][c][attribute] = vals_dict
                elif include_count and corpus_stats_dict:
                    result["corpora"][c][attribute] = corpus_stats_dict
                elif not include_count and corpus_stats_set:
                    result["corpora"][c][attribute] = sorted(corpus_stats_set)

                if incremental:
                    yield {f"progress_{progress_count}": c}
                    progress_count += 1

    if combined:
        for c in result["corpora"]:
            _merge_into(result["combined"], result["corpora"][c])
    else:
        del result["combined"]

    if ctx.common.cache and not all_cache:
        for c in corpus:
            if c not in cache_prefixes:
                cache_prefixes[c] = await utils.cache_prefix(ctx.cache, c)
            for attribute in attr:
                if (c, attribute) in from_cache:
                    continue
                checksum = utils.get_hash((c, attribute, split, include_count))
                try:
                    cache_key = f"{cache_prefixes[c]}:attr_values_{checksum}"
                    await ctx.cache.add(cache_key, result["corpora"][c].get(attribute, {}))
                except CacheError:
                    pass
                else:
                    if ctx.common.debug:
                        result.setdefault("DEBUG", {})
                        result["DEBUG"].setdefault("caches_saved", [])
                        result["DEBUG"]["caches_saved"].append(f"{c}:{attribute}")

    if not per_corpus:
        del result["corpora"]

    yield result


def _merge_into(target: dict, source: dict) -> None:
    """Merge source attribute value dict into target in place.

    Args:
        target: Target dict (modified in-place).
        source: Source dict to merge from.
    """
    for key, value in source.items():
        if key in target:
            if isinstance(target[key], dict) and isinstance(value, dict):
                _merge_into(target[key], value)
            elif isinstance(target[key], int):
                target[key] += value
            elif isinstance(target[key], list):
                target[key] = sorted(set(target[key] + value))
        else:
            target[key] = deepcopy(value)

"""Route for listing CWB attribute values."""

import itertools
from collections import defaultdict
from collections.abc import AsyncIterator
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Annotated, Any, TypeAlias

from pydantic import BeforeValidator, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

from korp.api import params, schemas

if TYPE_CHECKING:
    import anyio.abc
import anyio
from anyio import CapacityLimiter
from fastapi import APIRouter, Query

from korp import auth, caching, handler, utils
from korp.config import settings
from korp.dependencies import CtxDep
from korp.handler import api_handler
from korp.memcached import CacheError

from . import frequencies as frequencies_router

router = APIRouter(tags=["Corpus Information"])

ATTRIBUTE_VALUES_DESCRIPTION = """List the values available for one or more CWB attributes.

The route can be used for positional CWB attributes (i.e. token annotations) such as `word`, `lemma`, or `pos`, and
for structural CWB attributes (usually sentence or document annotations) such as `text_author` or `text_title`. It is
similar to `/frequencies/corpus`, but the result is organized as a lookup of attribute values instead of frequency rows,
and it supports hierarchical attribute expressions.

Use `attributes` to request one or more CWB attribute names. A value can be a single attribute, such as `text_author`,
or a hierarchy using `>`, such as `text_author>text_title`. Hierarchical expressions produce nested objects, which are
useful for building dependent filters: for example, authors as top-level keys and their titles as child values.

By default the result contains value lists. When `include_counts=true`, leaf values are token counts instead. Use
`split` for set-valued CWB attributes whose values should be split on `|` before being included in the result.

Use `combined` and `per_corpus` to choose whether to include merged values across all selected corpora, per-corpus
values, or both. When `incremental=true`, progress keys such as `progress_corpora` and `progress_0` may be included
before the final result in the streamed JSON object.

### Example

Get all authors and their titles with token counts:

`/attribute_values?corpus=ROMI&attributes=text_author>text_title&include_counts=true`
"""

AttrParam: TypeAlias = Annotated[
    list[str],
    Query(
        description=(
            "Comma-separated list of CWB attribute names or attribute hierarchies. Use `>` to request nested values, "
            "for example `text_author>text_title`."
        ),
        examples=[["text_author"], ["text_author>text_title"], ["pos,lemma"]],
    ),
    BeforeValidator(utils.split_csv),
]

AttrValuesSplitParam: TypeAlias = Annotated[
    list[str] | SkipJsonSchema[None],
    Query(
        description="Comma-separated list of set-valued CWB attributes to split on `|` before collecting values.",
        examples=[["text_topic"], ["sense,lemma"]],
    ),
    BeforeValidator(utils.split_csv),
]

IncludeCountsParam: TypeAlias = Annotated[
    bool,
    Query(
        description=(
            "Whether to return token counts for each leaf value. When disabled, leaf values are returned as lists; "
            "when enabled, leaf values are returned as objects mapping value to count."
        )
    ),
]

AttributeValuesData = dict[str, list[str] | dict[str, Any]]


class AttrValuesResponse(schemas.CommonResponse):
    """Response model for `/attribute_values` route."""

    model_config = ConfigDict(extra="allow")

    corpora: dict[str, AttributeValuesData] | SkipJsonSchema[None] = Field(
        None,
        description=(
            "Per-corpus CWB attribute values, keyed by corpus id. Omitted when `per_corpus=false`. Within each corpus, "
            "keys are the requested `attributes` expressions."
        ),
        examples=[{"ROMI": {"text_author": ["Söderberg, Hjalmar"], "pos": {"NN": 1250, "VB": 341}}}],
    )
    combined: AttributeValuesData | SkipJsonSchema[None] = Field(
        None,
        description=(
            "CWB attribute values merged across all selected corpora. Omitted when `combined=false`. Keys are the "
            "requested `attributes` expressions."
        ),
        examples=[{"text_author>text_title": {"Söderberg, Hjalmar": {"Doktor Glas": 12345}}}],
    )
    progress_corpora: list[str] | SkipJsonSchema[None] = Field(
        None,
        description=(
            "Corpora that will produce incremental progress updates. Included only when `incremental=true`; individual "
            "progress entries are returned as dynamic keys such as `progress_0`."
        ),
        examples=[["ROMI", "SUC3"]],
    )


@router.get(
    "/attribute_values",
    response_model=None,
    responses=handler.docs_response(AttrValuesResponse),
    summary="Attribute Values",
    description=ATTRIBUTE_VALUES_DESCRIPTION,
)
@router.post("/attribute_values", response_model=None, include_in_schema=False)
@api_handler
async def attribute_values(
    ctx: CtxDep,
    corpus: params.CorpusParam,
    attributes: AttrParam,
    include_counts: IncludeCountsParam = False,
    per_corpus: params.PerCorpusParam = True,
    combined: params.CombinedParam = True,
    split: AttrValuesSplitParam = None,
) -> AsyncIterator[dict]:
    """Get all available values for one or more corpus annotations.

    Args:
        ctx: Request context.
        corpus: Comma-separated list of corpora.
        attributes: Comma-separated list of CWB attribute names or attribute hierarchies.
        include_counts: Whether to include counts for each attribute value.
        per_corpus: Whether to include per-corpus results.
        combined: Whether to include combined results across corpora.
        split: Comma-separated list of CWB attributes to split values for.

    Yields:
        CWB attribute values (and counts, if requested) for the specified corpora and annotations.
    """
    incremental = ctx.common.incremental

    await auth.check_authorization(corpus, ctx)

    split = split or []
    split_set = set(split)
    result = {"corpora": defaultdict(dict), "combined": {}}
    from_cache = set()  # Keep track of what has been read from cache
    cache_prefixes: dict[str, str] = {}  # Reused between cache read and write phases

    if ctx.common.cache:
        all_cache = True
        for c in corpus:
            cache_prefixes[c] = await caching.cache_prefix(ctx.cache, c)
            for attribute in attributes:
                checksum = utils.get_hash((c, attribute, split, include_counts))
                data = await ctx.cache.get(f"{cache_prefixes[c]}:attribute_values_{checksum}")
                if data is not None:
                    result["corpora"][c][attribute] = data
                    if ctx.common.debug:
                        result.setdefault("debug", {"caches_read": []})
                        result["debug"]["caches_read"].append(f"{c}:{attribute}")
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
            """Worker function to run a frequency query in a thread."""
            async with send_channel:  # Closes the channel when done
                lines, _nr_hits, _corpus_size = await anyio.to_thread.run_sync(  # type: ignore
                    partial(  # Use partial to be able to pass keyword arguments
                        frequencies_router.simple_frequency_query_worker,
                        ctx=ctx,
                        corpus=corpus,
                        cqp_query=[],
                        group_by=[(s, True) for s in attr.split(">")],
                        use_cache=ctx.common.cache,
                    ),
                    limiter=limiter,
                )
                await send_channel.send((corpus, attr, lines))

        async with anyio.create_task_group() as tg:
            for c in corpus:
                for attribute in attributes:
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
                            if include_counts:
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
                            if include_counts:
                                corpus_stats_dict[v] = freq
                            else:
                                corpus_stats_set.add(v)

                if is_nested:
                    result["corpora"][c][attribute] = vals_dict
                elif include_counts and corpus_stats_dict:
                    result["corpora"][c][attribute] = corpus_stats_dict
                elif not include_counts and corpus_stats_set:
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
                cache_prefixes[c] = await caching.cache_prefix(ctx.cache, c)
            for attribute in attributes:
                if (c, attribute) in from_cache:
                    continue
                checksum = utils.get_hash((c, attribute, split, include_counts))
                try:
                    cache_key = f"{cache_prefixes[c]}:attribute_values_{checksum}"
                    await ctx.cache.add(cache_key, result["corpora"][c].get(attribute, {}))
                except CacheError:
                    pass
                else:
                    if ctx.common.debug:
                        result.setdefault("debug", {})
                        result["debug"].setdefault("caches_saved", [])
                        result["debug"]["caches_saved"].append(f"{c}:{attribute}")

    if not per_corpus:
        del result["corpora"]

    yield result


def _merge_into(target: dict, source: dict) -> None:
    """Merge a source annotation-value dictionary into a target in place.

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

"""Router for log-likelihood comparison."""

import asyncio
import dataclasses
import math
from collections import defaultdict
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import AfterValidator, BeforeValidator

from korp import utils
from korp.api import params

from . import count

router = APIRouter(tags=["Statistics"])


@router.get("/loglike", response_model=dict)
@router.post("/loglike", response_model=dict, include_in_schema=False)
@utils.api_handler
async def loglike(
    ctx: utils.CtxDep,
    set1_cqp: Annotated[str, Query(description="CQP query for set 1")],
    set2_cqp: Annotated[str, Query(description="CQP query for set 2")],
    set1_corpus: Annotated[
        list[str],
        Query(description="Comma-separated list of corpora for set 1"),
        BeforeValidator(utils.split_csv),
        AfterValidator(lambda v: [x.upper() for x in v]),
    ],
    set2_corpus: Annotated[
        list[str],
        Query(description="Comma-separated list of corpora for set 2"),
        BeforeValidator(utils.split_csv),
        AfterValidator(lambda v: [x.upper() for x in v]),
    ],
    max_results: Annotated[int, Query(description="Maximum number of results", alias="max")] = 15,
    group_by: count.GroupByParam = None,
    group_by_struct: count.GroupByStructParam = None,
    within: params.WithinParam = None,
    default_within: params.DefaultWithinParam = None,
    # cut: int | None = None,
    start: count.StartParam = 0,
    end: count.EndParam = -1,
    ignore_case: count.IgnoreCaseParam = None,
    relative_to_struct: count.RelativeToStructParam = None,
    split: params.SplitParam = None,
    strip_pointer: count.StripPointerParam = None,
    top: count.TopParam = None,
    expand_prequeries: params.ExpandPrequeriesParam = True,
    abort_signal: utils.AbortDep = None,
) -> AsyncGenerator[dict]:
    """Do a log-likelihood comparison on two queries.

    Yields:
        A dictionary with log-likelihood results.
    """
    # Handle parameters common to count
    params = await count.parse_parameters(
        ctx=ctx,
        corpus=[],
        cqp=[],
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
        simple=False,
        expand_prequeries=expand_prequeries,
        start=start,
        end=end,
    )

    # Handle parameters specific to loglike
    set1_corpora = set(set1_corpus)
    set2_corpora = set(set2_corpus)

    corpora = set1_corpora.union(set2_corpora)
    await utils.check_authorization(corpora, ctx)

    same_cqp = set1_cqp == set2_cqp

    def _make_freq_key(value: dict) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Create a hashable frequency key from a count result row's value dict.

        Args:
            value: A dict mapping attribute names to their values.

        Returns:
            A sorted tuple of (attr_name, values_tuple) pairs.
        """
        return tuple(
            (k, v if isinstance(v, tuple) else (v,))
            for k, v in sorted(value.items())
        )

    def compute_loglike(wf1: int, tot1: int, wf2: int, tot2: int) -> float:
        """Compute log-likelihood for a single pair.

        Args:
            wf1: Word frequency in set 1.
            tot1: Total word count in set 1.
            wf2: Word frequency in set 2.
            tot2: Total word count in set 2.

        Returns:
            Log-likelihood value.
        """
        # Expected frequencies
        wf_total = wf1 + wf2
        tot_total = tot1 + tot2
        e1 = wf_total * (tot1 / tot_total)
        e2 = wf_total * (tot2 / tot_total)

        (l1, l2) = (0, 0)
        if wf1 > 0:
            l1 = wf1 * math.log(wf1 / e1)
        if wf2 > 0:
            l2 = wf2 * math.log(wf2 / e2)
        loglike = 2 * (l1 + l2)
        return round(loglike, 2)

    def compute_list(d1: dict, tot1: int, ref: dict, reftot: int) -> list[tuple[float, str]]:
        """Compute log-likelihood for lists.

        Args:
            d1: Word frequency dictionary for set 1.
            tot1: Total word count for set 1.
            ref: Word frequency dictionary for set 2.
            reftot: Total word count for set 2.

        Returns:
            List of tuples (log-likelihood, word), sorted by log-likelihood descending.
        """
        all_words = d1.keys() | ref.keys()
        result = [
            (compute_loglike(d1.get(w, 0), tot1, ref.get(w, 0), reftot), w)
            for w in all_words
        ]
        result.sort(reverse=True)
        return result

    def compute_ll_stats(
        ll_list: list[tuple[float, str]],
        count: int,
        sets: list[dict],
    ) -> tuple[list[tuple[float, str]], float]:
        """Calculate average and truncate word list.

        Words more prominent in set 1 get a negated log-likelihood value.

        Args:
            ll_list: List of tuples (log-likelihood, word).
            count: Maximum number of words to include from each set. 0 means no limit.
            sets: List of two dictionaries with 'total' and 'freq' keys for each set.

        Returns:
            A tuple containing:
                - Truncated list of tuples (log-likelihood, word).
                - Average log-likelihood.
        """
        new_list = []
        set1count, set2count = 0, 0

        for ll, w in ll_list:
            freq1 = sets[0]["freq"].get(w, 0)
            freq2 = sets[1]["freq"].get(w, 0)
            in_set1 = freq1 and (not freq2 or freq1 / sets[0]["total"] > freq2 / sets[1]["total"])

            if in_set1:
                set1count += 1
                if not count or set1count <= count:
                    new_list.append((-ll, w))
            else:
                set2count += 1
                if not count or set2count <= count:
                    new_list.append((ll, w))

            if count and set1count >= count and set2count >= count:
                break

        avg = round(sum(ll for ll, _ in ll_list) / len(ll_list), 2) if ll_list else 0.0
        return new_list, avg

    result = {}

    # If same CQP for both sets, handle as one query for better performance
    if same_cqp:
        params.cqp = [set1_cqp]
        params.corpora = list(corpora)
        count_result = await utils.async_generator_to_dict(count.perform_count(params, ctx, abort_signal))

        sets = [{"total": 0, "freq": defaultdict(int)}, {"total": 0, "freq": defaultdict(int)}]
        for i, cset in enumerate((set1_corpora, set2_corpora)):
            for corpus in cset:
                sets[i]["total"] += count_result["corpora"][corpus]["sums"]["absolute"]
                if len(cset) == 1:
                    sets[i]["freq"] = {
                        _make_freq_key(x["value"]): x["absolute"]
                        for x in count_result["corpora"][corpus]["rows"]
                    }
                else:
                    for x in count_result["corpora"][corpus]["rows"]:
                        sets[i]["freq"][_make_freq_key(x["value"])] += x["absolute"]

    else:
        params_2 = dataclasses.replace(params)
        params.corpora = list(set1_corpora)
        params.cqp = [set1_cqp]
        params_2.corpora = list(set2_corpora)
        params_2.cqp = [set2_cqp]
        count_result_1, count_result_2 = await asyncio.gather(
            utils.async_generator_to_dict(count.perform_count(params, ctx, abort_signal)),
            utils.async_generator_to_dict(count.perform_count(params_2, ctx, abort_signal)),
        )

        sets = [{}, {}]
        for i, res in enumerate((count_result_1, count_result_2)):
            sets[i]["total"] = res["combined"]["sums"]["absolute"]
            sets[i]["freq"] = {
                _make_freq_key(row["value"]): row["absolute"]
                for row in res["combined"]["rows"]
            }

    ll_list = compute_list(sets[0]["freq"], sets[0]["total"], sets[1]["freq"], sets[1]["total"])
    ws, avg = compute_ll_stats(ll_list, max_results, sets)

    result["loglike"] = {}
    result["average"] = avg
    result["set1"] = {}
    result["set2"] = {}

    for ll, w in ws:
        w_formatted = " ".join(w[0][1])
        result["loglike"][w_formatted] = ll
        result["set1"][w_formatted] = sets[0]["freq"].get(w, 0)
        result["set2"][w_formatted] = sets[1]["freq"].get(w, 0)

    yield result

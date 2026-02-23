"""Router for log-likelihood comparison."""

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

router = APIRouter()


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
    params = count.parse_parameters(
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
    utils.check_authorization(corpora, ctx)

    same_cqp = set1_cqp == set2_cqp

    def expected(total: float, wordtotal: float, sumtotal: float) -> float:
        """Calculate expected frequency.

        The expectation is that the words are uniformly distributed over the corpora.

        Args:
            total: Total word count in the corpus.
            wordtotal: Total count of the word in both corpora.
            sumtotal: Total word count in both corpora.

        Returns:
            Expected frequency of the word in the corpus.
        """
        return wordtotal * (float(total) / sumtotal)

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
        e1 = expected(tot1, wf1 + wf2, tot1 + tot2)
        e2 = expected(tot2, wf1 + wf2, tot1 + tot2)
        (l1, l2) = (0, 0)
        if wf1 > 0:
            l1 = wf1 * math.log(wf1 / e1)
        if wf2 > 0:
            l2 = wf2 * math.log(wf2 / e2)
        loglike = 2 * (l1 + l2)
        return round(loglike, 2)

    def compute_list(d1: dict, tot1: int, ref: dict, reftot: int) -> list[tuple[float, str]]:
        """Compute log-likelyhood for lists.

        Args:
            d1: Word frequency dictionary for set 1.
            tot1: Total word count for set 1.
            ref: Word frequency dictionary for set 2.
            reftot: Total word count for set 2.

        Returns:
            List of tuples (log-likelihood, word), sorted by log-likelihood descending.
        """
        result = []
        # Get all words in either set
        all_w = set(d1.keys()).union(set(ref.keys()))
        for w in all_w:
            ll = compute_loglike(d1.get(w, 0), tot1, ref.get(w, 0), reftot)
            result.append((ll, w))
        result.sort(reverse=True)
        return result

    def compute_ll_stats(
        ll_list: list[tuple[float, str]],
        count: int,
        sets: list[dict],
    ) -> tuple[list[tuple[float, str]], float, float, float]:
        """Calculate max, min, average, and truncate word list.

        Args:
            ll_list: List of tuples (log-likelihood, word).
            count: Maximum number of words to include from each set.
            sets: List of two dictionaries with 'total' and 'freq' keys for each set.

        Returns:
            A tuple containing:
                - Truncated list of tuples (log-likelihood, word).
                - Average log-likelihood.
                - Minimum log-likelihood.
                - Maximum log-likelihood.
        """
        tot = len(ll_list)
        new_list = []

        set1count, set2count = 0, 0
        for ll_w in ll_list:
            ll, w = ll_w

            if (sets[0]["freq"].get(w) and not sets[1]["freq"].get(w)) or (
                sets[0]["freq"].get(w)
                and (sets[0]["freq"].get(w, 0) / (sets[0]["total"] * 1.0))
                > (sets[1]["freq"].get(w, 0) / (sets[1]["total"] * 1.0))
            ):
                set1count += 1
                if set1count <= count or not count:
                    new_list.append((ll * -1, w))
            else:
                set2count += 1
                if set2count <= count or not count:
                    new_list.append((ll, w))

            if count and (set1count >= count and set2count >= count):
                break

        nums = [ll for (ll, _) in ll_list]
        return (
            new_list,
            round(sum(nums) / float(tot), 2) if tot else 0.0,
            min(nums) if nums else 0.0,
            max(nums) if nums else 0.0,
        )

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
                        tuple(
                            (y[0], y[1] if isinstance(y[1], tuple) else (y[1],)) for y in sorted(x["value"].items())
                        ): x["absolute"]
                        for x in count_result["corpora"][corpus]["rows"]
                    }
                else:
                    for w, f in (
                        (
                            tuple(
                                (y[0], y[1] if isinstance(y[1], tuple) else (y[1],)) for y in sorted(x["value"].items())
                            ),
                            x["absolute"],
                        )
                        for x in count_result["corpora"][corpus]["rows"]
                    ):
                        sets[i]["freq"][w] += f

    else:
        params_2 = dataclasses.replace(params)
        params.corpora = list(set1_corpora)
        params.cqp = [set1_cqp]
        params_2.corpora = list(set2_corpora)
        params_2.cqp = [set2_cqp]
        count_result = [
            await utils.async_generator_to_dict(count.perform_count(params, ctx, abort_signal)),
            await utils.async_generator_to_dict(count.perform_count(params_2, ctx, abort_signal)),
        ]

        sets = [{}, {}]
        for i, res in enumerate(count_result):
            sets[i]["total"] = res["combined"]["sums"]["absolute"]
            sets[i]["freq"] = {
                tuple((y[0], y[1] if isinstance(y[1], tuple) else (y[1],)) for y in sorted(row["value"].items())): row[
                    "absolute"
                ]
                for row in res["combined"]["rows"]
            }

    ll_list = compute_list(sets[0]["freq"], sets[0]["total"], sets[1]["freq"], sets[1]["total"])
    (ws, avg, _mi, _ma) = compute_ll_stats(ll_list, max_results, sets)

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

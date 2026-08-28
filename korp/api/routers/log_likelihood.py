"""Router for log-likelihood comparison."""

import asyncio
import dataclasses
import math
from collections import defaultdict
from collections.abc import AsyncGenerator
from typing import Annotated, TypeAlias

from fastapi import APIRouter, Query
from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field

from korp import auth, utils
from korp.api import params, schemas
from korp.dependencies import AbortDep, CtxDep
from korp.handler import api_handler, docs_response

from . import frequencies

router = APIRouter(tags=["Statistics"])

LOGLIKE_DESCRIPTION = """Compare two searches with log-likelihood.

The route first counts the requested CWB attribute values in two sets, then calculates a log-likelihood score for each
value. By default, values are grouped by `word`; use `group_by` and `group_by_struct` to compare other positional or
structural CWB attributes.

The sign of each score shows which set the value is relatively more prominent in:

- Negative values are more prominent in set 1.
- Positive values are more prominent in set 2.
- Larger absolute values indicate a stronger difference between the sets.

Each result row contains the grouped value, its log-likelihood score, and the absolute frequencies used from each set.
`average` is the average absolute log-likelihood score before the result list is split by sign and limited.

Use `max_results` to limit how many values to return from each side of the comparison. For example, `max_results=10` can
return up to ten set-1-prominent values and ten set-2-prominent values. Use `max_results=0` for no limit.

Most grouping and value-normalization parameters are shared with `/frequencies`, including `group_by`,
`group_by_struct`, `ignore_case`, `split`, `strip_pointer`, `top`, `within`, and `default_within`.

### Example

Compare nouns in two corpora and return up to ten values from each side:

`/log_likelihood?set1_cqp=[pos="NN"]&set2_cqp=[pos="NN"]&group_by=word&max_results=10&set1_corpora=ROMI&set2_corpora=GP2012`
"""

Set1CQPParam: TypeAlias = Annotated[
    str,
    Query(
        description="CQP query for set 1.",
        examples=['[pos="NN"]'],
    ),
]

Set2CQPParam: TypeAlias = Annotated[
    str,
    Query(
        description="CQP query for set 2.",
        examples=['[pos="NN"]'],
    ),
]

Set1CorporaParam: TypeAlias = Annotated[
    list[str],
    Query(description="Comma-separated list of corpora for set 1.", examples=[["ROMI,SUC3"]]),
    BeforeValidator(utils.split_csv),
    AfterValidator(lambda v: [x.upper() for x in v]),
]

Set2CorporaParam: TypeAlias = Annotated[
    list[str],
    Query(description="Comma-separated list of corpora for set 2.", examples=[["GP2012"]]),
    BeforeValidator(utils.split_csv),
    AfterValidator(lambda v: [x.upper() for x in v]),
]

MaxResultsParam: TypeAlias = Annotated[
    int,
    Query(
        description=("Maximum number of results to return from each side of the comparison. Use 0 for no limit."),
        ge=0,
        examples=[15],
    ),
]


class LogLikelihoodRow(BaseModel):
    """A log-likelihood result row."""

    value: str = Field(..., description="Grouped value being compared.", examples=["cat"])
    score: float = Field(
        ...,
        description=(
            "Log-likelihood score. Negative values are more prominent in set 1; positive values are more prominent "
            "in set 2."
        ),
        examples=[-5.43],
    )
    set1: int = Field(..., description="Absolute frequency in set 1.", examples=[447])
    set2: int = Field(..., description="Absolute frequency in set 2.", examples=[254])


class LogLikelihoodResponse(schemas.CommonResponse):
    """Response model for `/log_likelihood` route."""

    model_config = ConfigDict(extra="allow")

    average: float = Field(
        ...,
        description="Average absolute log-likelihood score across all compared values before result limiting.",
        examples=[12.43],
    )
    results: list[LogLikelihoodRow] = Field(
        ...,
        description="Log-likelihood rows for the returned values.",
        examples=[[{"value": "cat", "score": -5.43, "set1": 447, "set2": 254}]],
    )


@router.get(
    "/log_likelihood",
    response_model=None,
    responses=docs_response(LogLikelihoodResponse),
    summary="Log-Likelihood Comparison",
    description=LOGLIKE_DESCRIPTION,
)
@router.post("/log_likelihood", response_model=None, include_in_schema=False)
@api_handler
async def log_likelihood(
    ctx: CtxDep,
    set1_cqp: Set1CQPParam,
    set2_cqp: Set2CQPParam,
    set1_corpora: Set1CorporaParam,
    set2_corpora: Set2CorporaParam,
    max_results: MaxResultsParam = 15,
    group_by: frequencies.GroupByParam = None,
    group_by_struct: frequencies.GroupByStructParam = None,
    within: params.WithinParam = None,
    default_within: params.DefaultWithinParam = None,
    # cut: int | None = None,
    offset: frequencies.OffsetParam = 0,
    limit: frequencies.LimitParam = 0,
    ignore_case: frequencies.IgnoreCaseParam = None,
    relative_to_struct: frequencies.RelativeToStructParam = None,
    split: params.SplitParam = None,
    strip_pointer: frequencies.StripPointerParam = None,
    top: frequencies.TopParam = None,
    expand_prequeries: params.ExpandPrequeriesParam = True,
    abort_signal: AbortDep = None,
) -> AsyncGenerator[dict]:
    """Do a log-likelihood comparison on two queries.

    Yields:
        A dictionary with log-likelihood results.
    """
    # Handle parameters common to frequency queries
    frequency_params = await frequencies.parse_frequency_parameters(
        ctx=ctx,
        corpora=[],
        cqp_query=[],
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
        offset=offset,
        limit=limit,
    )

    # Handle parameters specific to log-likelihood
    set1_corpora_set = set(set1_corpora)
    set2_corpora_set = set(set2_corpora)

    corpora = set1_corpora_set.union(set2_corpora_set)
    await auth.check_authorization(corpora, ctx)

    same_cqp = set1_cqp == set2_cqp

    def _make_freq_key(value: dict) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Create a hashable frequency key from a frequency result row's value dict.

        Args:
            value: A dict mapping CWB attribute names to their values.

        Returns:
            A sorted tuple of (attr_name, values_tuple) pairs.
        """
        return tuple((k, v if isinstance(v, tuple) else (v,)) for k, v in sorted(value.items()))

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
        result = [(compute_loglike(d1.get(w, 0), tot1, ref.get(w, 0), reftot), w) for w in all_words]
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
        frequency_params.cqp_query = [set1_cqp]
        frequency_params.corpora = list(corpora)
        frequency_result = await utils.async_generator_to_dict(
            frequencies.perform_frequency_query(frequency_params, ctx, abort_signal)
        )

        sets = [{"total": 0, "freq": defaultdict(int)}, {"total": 0, "freq": defaultdict(int)}]
        for i, cset in enumerate((set1_corpora, set2_corpora)):
            for corpus in cset:
                sets[i]["total"] += frequency_result["corpora"][corpus]["sums"]["absolute"]
                if len(cset) == 1:
                    sets[i]["freq"] = {
                        _make_freq_key(x["value"]): x["absolute"] for x in frequency_result["corpora"][corpus]["rows"]
                    }
                else:
                    for x in frequency_result["corpora"][corpus]["rows"]:
                        sets[i]["freq"][_make_freq_key(x["value"])] += x["absolute"]

    else:
        frequency_params_2 = dataclasses.replace(frequency_params)
        frequency_params.corpora = list(set1_corpora)
        frequency_params.cqp_query = [set1_cqp]
        frequency_params_2.corpora = list(set2_corpora)
        frequency_params_2.cqp_query = [set2_cqp]
        frequency_result_1, frequency_result_2 = await asyncio.gather(
            utils.async_generator_to_dict(
                frequencies.perform_frequency_query(frequency_params, ctx, abort_signal)
            ),
            utils.async_generator_to_dict(
                frequencies.perform_frequency_query(frequency_params_2, ctx, abort_signal)
            ),
        )

        sets = [{}, {}]
        for i, res in enumerate((frequency_result_1, frequency_result_2)):
            sets[i]["total"] = res["combined"]["sums"]["absolute"]
            sets[i]["freq"] = {_make_freq_key(row["value"]): row["absolute"] for row in res["combined"]["rows"]}

    ll_list = compute_list(sets[0]["freq"], sets[0]["total"], sets[1]["freq"], sets[1]["total"])
    ws, avg = compute_ll_stats(ll_list, max_results, sets)

    result["average"] = avg
    result["results"] = []

    for ll, w in ws:
        w_formatted = " ".join(w[0][1])
        result["results"].append(
            {
                "value": w_formatted,
                "score": ll,
                "set1": sets[0]["freq"].get(w, 0),
                "set2": sets[1]["freq"].get(w, 0),
            }
        )

    yield result

"""Common parameter type aliases for Korp routes."""

from enum import StrEnum
from typing import Annotated, TypeAlias

from fastapi import Query
from pydantic import AfterValidator, BeforeValidator
from pydantic.json_schema import SkipJsonSchema

from korp import utils

CorporaParam: TypeAlias = Annotated[
    list[str],
    Query(
        description=(
            "Comma-separated list of corpus ids to query. Corpus ids are case-insensitive."
        ),
        examples=[["ROMI", "SUC3"]],
    ),
    BeforeValidator(utils.split_csv),
    AfterValidator(lambda v: sorted({x.strip().upper() for x in v})),
]

CQPParam: TypeAlias = Annotated[
    list[str],
    Query(
        description=(
            "CQP query or queries to run. Repeat the `cqp` parameter to perform several queries in sequence; each "
            "query is executed on the result of the previous query, in the order it appears in the request."
        ),
        alias="cqp",
        examples=[['[word="flower"]'], ['[word="flower"]', '[pos="NN"]']],
    ),
]

DefaultWithinParam: TypeAlias = Annotated[
    str | SkipJsonSchema[None],
    Query(
        description=(
            "Default structural unit that query matches must stay inside, for example `sentence` or `text`. This "
            "prevents matches and expanded prequery results from crossing structural boundaries."
        ),
        examples=["sentence"],
    ),
]

WithinParam: TypeAlias = Annotated[
    list[str] | SkipJsonSchema[None],
    Query(
        description=(
            "Per-corpus structural unit that query matches must stay inside, overriding `default_within` for the "
            "specified corpora. Use `CORPUS:structure`; multiple overrides can be comma-separated."
        ),
        examples=[["ROMI:paragraph,SUC3:sentence"]],
    ),
    BeforeValidator(utils.split_csv),
]

DefaultContextParam: TypeAlias = Annotated[
    str | SkipJsonSchema[None],
    Query(
        description=(
            "Default context to return around each match. Use `<number> <unit>`, for example `10 word` for token "
            "context or `1 sentence` for structural context."
        ),
        pattern=r"^\d+\s+[\w_-]+$",
        examples=["10 word", "1 sentence", "1 paragraph"],
    ),
]

ContextParam: TypeAlias = Annotated[
    list[str] | SkipJsonSchema[None],
    Query(
        description=(
            "Per-corpus context to return around each match, overriding `default_context` for the specified corpora. "
            "Use `CORPUS:<number> <unit>`; multiple overrides can be comma-separated."
        ),
        examples=[["ROMI:1 sentence,SUC3:10 word"]],
    ),
    BeforeValidator(utils.split_csv),
]

ExpandPrequeriesParam: TypeAlias = Annotated[
    bool,
    Query(
        description=(
            "When several `cqp` parameters are supplied, choose whether each intermediate result should be expanded "
            "to its containing `within` context before the next query is executed. Enabled by default; disabling it "
            "runs the next query only on the exact tokens matched by the previous query."
        )
    ),
]


class GranularityValues(StrEnum):
    """Allowed granularities for time-related routes."""

    year = "year"
    month = "month"
    day = "day"
    hour = "hour"
    minute = "minute"
    second = "second"


GranularityParam: TypeAlias = Annotated[
    GranularityValues,
    Query(
        description=(
            "Time resolution for returned buckets: `year`, `month`, `day`, `hour`, `minute`, or `second`."
        )
    ),
]


TIME_STRATEGY_DESCRIPTION = """The `strategy` parameter controls how dated material is matched to requested time spans
and to the result buckets created by `granularity`. It affects both date filtering and which tokens contribute to each
time-series data point.

This matters when the material is dated more coarsely than the requested span. For example, if a text is dated only as
`2005`, should it contribute to a query limited to `2005-01-01` through `2005-01-31`?

In the rules below, the material time span is the date span stored for a text. The result time span is either the
`date_from`/`date_to` filter or one generated result bucket, such as all of `2015` for `granularity=year` or January
2015 for `granularity=month`.

Let `t1` and `t2` be the start and end of the material time span, and `r1` and `r2` the start and end of the result time
span:

- `some_overlaps`: include material when either span fully contains the other:
  `(t1 >= r1 AND t2 <= r2) OR (t1 <= r1 AND t2 >= r2)`.
- `all_overlaps`: include material when the spans overlap in any way: `t1 <= r2 AND t2 >= r1`.
- `strict`: include material only when the material span is fully contained by the result span:
  `t1 >= r1 AND t2 <= r2`.
"""


class StrategyValues(StrEnum):
    """Allowed strategies for timespan matching."""

    some_overlaps = "some_overlaps"
    all_overlaps = "all_overlaps"
    strict = "strict"


StrategyParam: TypeAlias = Annotated[
    StrategyValues,
    Query(
        description=(
            "Date-span matching strategy for time-based routes. `some_overlaps` includes material when either the "
            "material span contains the result period or the result period contains the material span; `all_overlaps` "
            "includes any material span that overlaps the result period; `strict` includes only material spans "
            "contained by the result period."
        )
    ),
]

IncludeCombinedParam: TypeAlias = Annotated[
    bool,
    Query(description="Whether to include results merged across all selected corpora in the `combined` field."),
]
IncludePerCorpusParam: TypeAlias = Annotated[
    bool,
    Query(description="Whether to include separate results for each selected corpus in the `corpora` field."),
]

SplitParam: TypeAlias = Annotated[
    list[str] | SkipJsonSchema[None],
    Query(
        description=(
            "Comma-separated list of set-valued CWB attributes whose values should be split on `|` before counting. "
            "Each split value is treated as a separate value in the result."
        ),
        examples=[["sense"], ["text_topic,sense"]],
    ),
    BeforeValidator(utils.split_csv),
]

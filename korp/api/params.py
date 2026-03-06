"""Common parameter type aliases for Korp routes."""

from enum import IntEnum, StrEnum
from typing import Annotated, TypeAlias

from fastapi import Query
from pydantic import AfterValidator, BeforeValidator
from pydantic.json_schema import SkipJsonSchema

from korp import utils

CorpusParam: TypeAlias = Annotated[
    list[str],
    Query(description="Comma-separated list of corpora."),
    BeforeValidator(utils.split_csv),
    AfterValidator(lambda v: sorted({x.strip().upper() for x in v})),
]

CQPParam: TypeAlias = Annotated[
    list[str],
    Query(description="CQP query or queries to perform.", alias="cqp"),
]

DefaultWithinParam: TypeAlias = Annotated[
    str | SkipJsonSchema[None],
    Query(
        description="The structural context to limit the search to, preventing matches from crossing structural "
        "boundaries."
    ),
]

WithinParam: TypeAlias = Annotated[
    list[str] | SkipJsonSchema[None],
    Query(
        description="The structural context to limit the search to, preventing matches "
        "from crossing structural boundaries. This overrides 'default_within' for the specified corpora. "
        "Format: `corpus1:struct1,corpus2:struct2,...`"
    ),
    BeforeValidator(utils.split_csv),
]

DefaultContextParam: TypeAlias = Annotated[
    str | SkipJsonSchema[None],
    Query(
        description="The default amount of context to show around each match, e.g., `10 word` or `2 sentence`.",
        pattern=r"^\d+\s+[\w_-]+$",
    ),
]

ContextParam: TypeAlias = Annotated[
    list[str] | SkipJsonSchema[None],
    Query(
        description="The amount of context to show around each match. This overrides 'default_context' for the "
        "specified corpora. Format: `corpus1:context1,corpus2:context2,...`"
    ),
    BeforeValidator(utils.split_csv),
]

ExpandPrequeriesParam: TypeAlias = Annotated[
    bool, Query(description="Whether to expand prequeries when using multiple CQP queries.")
]


class GranularityValues(StrEnum):
    """Allowed granularities for time-related routes."""

    year = "y"
    month = "m"
    day = "d"
    hour = "h"
    minute = "n"
    second = "s"


GranularityParam: TypeAlias = Annotated[
    GranularityValues,
    Query(
        description="Resolution of result (`y` = year, `m` = month, `d` = day, `h` = hour, `n` = minute, `s` = second)."
    ),
]


class StrategyValues(IntEnum):
    """Allowed strategies for timespan matching."""

    some_overlaps = 1
    all_overlaps = 2
    strict = 3


StrategyParam: TypeAlias = Annotated[
    StrategyValues,
    Query(
        description="Strategy for date range matching (1 = some overlaps permitted, 2 = all overlaps permitted, "
        "3 = strict matching)."
    ),
]

CombinedParam: TypeAlias = Annotated[bool, Query(description="Whether to include combined results.")]
PerCorpusParam: TypeAlias = Annotated[bool, Query(description="Whether to include results per corpus.")]

SplitParam: TypeAlias = Annotated[
    list[str] | SkipJsonSchema[None],
    Query(description="Attributes whose values should be split into separate hits."),
    BeforeValidator(utils.split_csv),
]

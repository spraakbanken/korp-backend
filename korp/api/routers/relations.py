"""Routes for Word Picture queries."""

import asyncio
import math
import operator
import time
from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Container, Mapping, Sequence
from copy import deepcopy
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias

from fastapi import APIRouter, Query
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from korp import auth, caching, utils
from korp.api import params, schemas
from korp.api.routers import info
from korp.config import settings
from korp.dependencies import AbortDep, AbortSignal, CtxDep
from korp.handler import api_handler, docs_response
from korp.memcached import CacheError

from . import query, timespan

router = APIRouter(tags=["Word Relations"])


# (role, head_string, head_pos, rel, dep_string, dep_pos, dep_extra)
RelationKey = tuple[str, str, str, str, str, str, str]


RMI_MULTIPLIER = 1_000
FREQ_RELATIVE_MULTIPLIER = 1_000_000
SPLIT_SUFFIX = "_yearly"


class RelationType(StrEnum):
    """Relation lookup type."""

    word = "word"
    lexeme = "lexeme"


class RelationsSort(StrEnum):
    """Allowed sort fields for relations."""

    freq = "freq"
    freq_relative = "freq_relative"
    mi = "mi"
    rmi = "rmi"


class PeriodAlign(StrEnum):
    """Allowed period alignment values."""

    oldest = "oldest"
    newest = "newest"


class MaxScope(StrEnum):
    """How max-result limiting is scoped."""

    overall = "overall"
    per_period = "per_period"


class Measures(StrEnum):
    """Available types of measures to include in the response."""

    freq = "freq"
    freq_relative = "freq_relative"
    mi = "mi"
    rmi = "rmi"


RELATIONS_DESCRIPTION = """Get word-picture dependency relations for a word or lexeme.

The route looks up dependency relations where the requested value occurs as either the head or the dependent. Each
relation row identifies the head, dependency relation, dependent, part-of-speech tags, optional dependent prefix, and
source ids that can be passed to `/relations_sentences`.

The statistical measures included in each row are controlled by `measures`. `freq` is the absolute relation frequency,
`freq_relative` is the relation frequency per million tokens, `mi` is lexicographer's mutual information, and `rmi` is
relative MI.

By default `/relations` returns overall relation statistics. Set `split=true` to also include time-sliced data in
`relations_time`; use `/relations_time` when you only want the time-sliced view.

### Example

Get dependency relations for the lexeme `ge..vb.1`:

`/relations?term=ge..vb.1&type=lexeme&corpus=ROMI`
"""

RELATIONS_TIME_DESCRIPTION = """Get word-picture dependency relations grouped by year or multi-year period.

The response groups rows in `relations_time` by period key. With `period_size=1`, keys are years such as `2018`; with
larger periods, keys are ranges such as `2016-2018`. Undated material is grouped under an empty key.

Use `period_size` and `period_align` to control how years are grouped. `max_scope=per_period` applies the `max` limit
inside each period; `max_scope=overall` first selects the top overall relations and then returns time data only for
those relations.
"""

RELATIONS_SENTENCES_DESCRIPTION = """Return KWIC sentences containing word-picture relation sources.

Use the `source` ids returned by `/relations` to retrieve the corpus sentences where those relation instances occur.
The sentence rows use the same KWIC structure as `/query`, with the relation span highlighted as the match.
"""

RELATIONS_TIME_SENTENCES_DESCRIPTION = """Return KWIC sentences for time-sliced word-picture relation sources.

This is the sentence lookup companion to `/relations_time`. It returns the same KWIC-style structure as
`/relations_sentences`.
"""

TermParam: TypeAlias = Annotated[
    str,
    Query(
        description="Word form or lexeme to look up. Use `type=lexeme` when the value is a lexeme.",
        examples=["ge..vb.1", "är"],
    ),
]

RelationTypeParam: TypeAlias = Annotated[
    RelationType,
    Query(
        alias="type",
        description="Interpret `term` as a plain word form or as a lexeme.",
    ),
]

MinFreqParam: TypeAlias = Annotated[
    int | None,
    Query(
        alias="min",
        ge=0,
        description="Minimum absolute relation frequency. Omit the parameter to use no frequency cutoff.",
        examples=[5],
    ),
]

MaxResultsParam: TypeAlias = Annotated[
    int,
    Query(
        alias="max",
        ge=0,
        description=("Maximum number of rows to return for each relation label and direction. Use `0` for no limit."),
        examples=[15],
    ),
]

RelationsSortParam: TypeAlias = Annotated[
    RelationsSort,
    Query(description="Measure used for sorting and for selecting rows when `max` applies."),
]

RelationsSplitParam: TypeAlias = Annotated[
    bool,
    Query(description="Whether `/relations` should include time-sliced results in `relations_time`."),
]

PeriodSizeParam: TypeAlias = Annotated[
    int,
    Query(
        ge=1,
        description="Number of years per time bucket. Use `1` for yearly output.",
        examples=[1, 5],
    ),
]

PeriodAlignParam: TypeAlias = Annotated[
    PeriodAlign,
    Query(
        description=(
            "How multi-year buckets are aligned when the available range is not evenly divisible by `period_size`: "
            "`newest` anchors buckets to the newest year, while `oldest` anchors them to the oldest year."
        )
    ),
]

YearParam: TypeAlias = Annotated[
    int | None,
    Query(
        ge=0,
        description="Inclusive year filter for time-sliced relation data.",
        examples=[2018],
    ),
]

OverallParam: TypeAlias = Annotated[
    bool,
    Query(description="Whether to include overall relation rows in `relations` in addition to time-sliced data."),
]

MaxScopeParam: TypeAlias = Annotated[
    MaxScope,
    Query(
        description=(
            "How `max` is applied when time-sliced data is requested: `per_period` selects top rows independently "
            "inside each period, while `overall` selects top overall rows first."
        )
    ),
]

MeasuresParam: TypeAlias = Annotated[
    Sequence[Measures],
    Query(
        description=(
            "Comma-separated list of measures to include on each relation row. The relation identifiers and `source` "
            "are always included."
        ),
        examples=[["freq,mi"], ["freq,freq_relative,mi,rmi"]],
    ),
    BeforeValidator(utils.split_csv),
]

RelationsOffsetParam: TypeAlias = Annotated[
    int,
    Query(description="Number of sentence rows to skip before returning results.", ge=0, examples=[0]),
]

RelationsLimitParam: TypeAlias = Annotated[
    int,
    Query(description="Maximum number of sentence rows to return.", ge=1, examples=[10]),
]

RelationsShowParam: TypeAlias = Annotated[
    str,
    Query(
        description="Comma-separated list of positional attributes to include on each returned token.",
        examples=["word,lemma,pos"],
    ),
]

RelationsShowStructParam: TypeAlias = Annotated[
    str,
    Query(
        description=(
            "Comma-separated list of structural attributes to include for each KWIC row. `sentence_id` is included by "
            "default."
        ),
        examples=["text_title,text_author"],
    ),
]

RelationsDefaultContextParam: TypeAlias = Annotated[
    str,
    Query(
        description="Context size for sentence lookup results.",
        examples=["1 sentence"],
    ),
]


class RelationRow(BaseModel):
    """A word-picture relation row."""

    head: str = Field(..., description="Head word form or lexeme.", examples=["cat"])
    headpos: str = Field(..., description="Part of speech for the head.", examples=["NN"])
    rel: str = Field(..., description="Dependency relation label.", examples=["AT"])
    dep: str = Field(..., description="Dependent word form or lexeme.", examples=["black"])
    deppos: str = Field(..., description="Part of speech for the dependent.", examples=["JJ"])
    depextra: str = Field(..., description="Dependent prefix or extra string data.", examples=[""])
    source: list[str] = Field(
        ...,
        description=(
            "Source ids for retrieving example sentences. Use `/relations_sentences` for overall relation rows and "
            "`/relations_time_sentences` for time-sliced relation rows."
        ),
        examples=[["ROMI:253662"]],
    )
    freq: int | SkipJsonSchema[None] = Field(None, description="Absolute relation frequency.", examples=[5])
    freq_relative: float | SkipJsonSchema[None] = Field(
        None,
        description="Relation frequency per one million corpus tokens.",
        examples=[2.13],
    )
    mi: float | SkipJsonSchema[None] = Field(
        None,
        description="Lexicographer's mutual information score.",
        examples=[17.326],
    )
    rmi: float | SkipJsonSchema[None] = Field(None, description="Relative MI score.", examples=[0.91])


class RelationsRange(BaseModel):
    """Year range covered by time-sliced relation output."""

    start: int = Field(..., description="First year covered by the returned time range.", examples=[2010])
    end: int = Field(..., description="Last year covered by the returned time range.", examples=[2020])


class RelationsResponse(schemas.CommonResponse):
    """Response model for `/relations` and `/relations_time` routes."""

    model_config = ConfigDict(extra="allow")

    relations: list[RelationRow] | SkipJsonSchema[None] = Field(
        None,
        description="Overall relation rows. Included when overall relation output is requested.",
    )
    relations_time: dict[str, list[RelationRow]] | SkipJsonSchema[None] = Field(
        None,
        description=(
            "Time-sliced relation rows keyed by year or period. Included when `split=true` on `/relations` or when "
            "using `/relations_time`."
        ),
    )
    token_frequencies: dict[str, int] | SkipJsonSchema[None] = Field(
        None,
        description="Total token frequencies for the same time buckets as `relations_time`.",
        examples=[{"2017": 15366, "2018": 7437}],
    )
    range: RelationsRange | SkipJsonSchema[None] = Field(
        None,
        description="Dated year range covered by the time-sliced output.",
    )
    period_size: int | SkipJsonSchema[None] = Field(
        None,
        description="Number of years represented by each period key in `relations_time`.",
        examples=[1],
    )
    progress_corpora: list[str] | SkipJsonSchema[None] = Field(
        None,
        description=(
            "Corpora that will produce incremental progress updates. Included only when `incremental=true`; individual "
            "progress entries are returned as dynamic keys such as `progress_0`."
        ),
        examples=[["ROMI", "SUC3"]],
    )


class RelationsSentencesResponse(schemas.CommonResponse):
    """Response model for relation sentence lookup routes."""

    model_config = ConfigDict(extra="allow")

    hits: int | SkipJsonSchema[None] = Field(None, description="Total number of matching relation sentences.")
    corpus_hits: dict[str, int] | SkipJsonSchema[None] = Field(
        None,
        description="Number of matching relation sentences per corpus.",
        examples=[{"ROMI": 3}],
    )
    corpus_order: list[str] | SkipJsonSchema[None] = Field(
        None,
        description="Order in which corpora are represented in the KWIC rows.",
        examples=[["ROMI"]],
    )
    kwic: list[query.KWICRow] | SkipJsonSchema[None] = Field(
        None,
        description="KWIC sentence rows using the same row structure as `/query`.",
    )


def _calc_freq_relative(freq: int, corpus_size: int) -> float:
    """Return frequency per million tokens."""
    if corpus_size <= 0:
        return 0.0
    return FREQ_RELATIVE_MULTIPLIER * freq / corpus_size


def _calc_rmi(mi_value: float, rel_freq: int) -> float:
    """Return relative MI."""
    if rel_freq <= 0:
        return 0.0
    return RMI_MULTIPLIER * mi_value / rel_freq


def _calc_mi(freq: int, head_rel_freq: int, dep_rel_freq: int, rel_freq: int) -> float:
    """Calculate mutual information value.

    Args:
        freq: The frequency of the relation triple.
        head_rel_freq: The frequency of the head-relation pair.
        dep_rel_freq: The frequency of the dep-relation pair.
        rel_freq: The total frequency of the relation.

    Returns:
        The MI value, or 0.0 if any frequency is zero.
    """
    if freq <= 0 or head_rel_freq <= 0 or dep_rel_freq <= 0 or rel_freq <= 0:
        return 0.0
    return freq * math.log2((rel_freq * freq) / (head_rel_freq * dep_rel_freq))


def _relation_output(entry: dict, measures: Container[Measures]) -> dict[str, str | int | float]:
    """Build the standard relation output dictionary.

    Args:
        entry: Dictionary containing relation data.
        measures: Set of measures to include in the output.

    Returns:
        Dictionary with the standard output fields.
    """
    output: dict[str, str | int | float] = {
        "head": entry["head"],
        "headpos": entry["headpos"],
        "rel": entry["rel"],
        "dep": entry["dep"],
        "deppos": entry["deppos"],
        "depextra": entry["depextra"],
        "source": entry["source"],
    }
    if Measures.freq in measures:
        output["freq"] = entry["freq"]
    if Measures.freq_relative in measures:
        output["freq_relative"] = entry["freq_relative"]
    if Measures.mi in measures:
        output["mi"] = entry["mi"]
    if Measures.rmi in measures:
        output["rmi"] = entry["rmi"]
    return output


def _table_names(corpus: str, *, split: bool) -> dict[str, str]:
    """Return the table names for a corpus.

    Args:
        corpus: Corpus identifier.
        split: If `True`, return the per-year (split) table names; otherwise overall tables.
    """
    prefix = f"{settings.DB_WP_TABLE}_{corpus.upper()}"
    main_prefix = f"{prefix}{SPLIT_SUFFIX}" if split else prefix
    return {
        "strings": f"{prefix}_strings",  # Shared between split and overall
        "main": main_prefix,
        "head_rel": f"{main_prefix}_head_rel",
        "dep_rel": f"{main_prefix}_dep_rel",
        "rel": f"{main_prefix}_rel",
    }


def _lexeme_clause(lexeme: bool, *, second: bool = False) -> str:
    """Return SQL clause for lexeme or word form selection.

    Args:
        lexeme: If `True`, select lexemes; if `False`, select word forms.
        second: If `True`, generate clause for the second query (dep); if `False`, for the first (head).

    Returns:
        SQL clause as a string.
    """
    if lexeme:
        return " f.bfhead = 1 AND f.bfdep = 1"
    if second:
        return " f.wfdep = 1"
    return " f.wfhead = 1"


def _year_clause(
    column: str,
    start_year: int | None,
    end_year: int | None,
    *,
    prefix: str,
) -> tuple[str, dict[str, int]]:
    """Return SQL clause for year range selection.

    Args:
        column: The column name to apply the year constraints to.
        start_year: The start year (inclusive) or `None`.
        end_year: The end year (inclusive) or `None`.
        prefix: Prefix for parameter names to avoid collisions.

    Returns:
        A tuple containing the SQL clause as a string and a dictionary of parameters.
    """
    clauses: list[str] = []
    params: dict[str, int] = {}
    if start_year is not None:
        param_name = f"{prefix}_start_year"
        clauses.append(f"{column} >= :{param_name}")
        params[param_name] = start_year
    if end_year is not None:
        param_name = f"{prefix}_end_year"
        clauses.append(f"{column} <= :{param_name}")
        params[param_name] = end_year
    if not clauses:
        return "", params
    combined = " AND ".join(clauses)
    return f"({column} IS NULL OR ({combined}))", params


def _build_overall_triples_query(
    corpus: str,
    *,
    lexeme: bool,
    min_freq: int | None = None,
) -> tuple[str, dict[str, object]]:
    """Build combined SQL query for overall relations (no year splitting).

    Args:
        corpus: The corpus name.
        lexeme: If `True`, select lexemes; if `False`, select word forms.
        min_freq: Minimum frequency filter or `None` (no minimum).

    Returns:
        A tuple containing the SQL query as a string and a dictionary of parameters.
    """
    tables = _table_names(corpus, split=False)
    strings = tables["strings"]
    main = tables["main"]
    rel_table = tables["rel"]
    head_rel = tables["head_rel"]
    dep_rel = tables["dep_rel"]
    corpus_label = utils.sql_escape(corpus.upper())
    freq_clause = ""
    params: dict[str, object] = {}
    if min_freq is not None:
        freq_clause = " AND f.freq >= :min_freq"
        params["min_freq"] = min_freq

    def _role_select(role: Literal["head", "dep"]) -> str:
        """Build one half of the UNION ALL for a given role.

        Returns:
            SQL SELECT statement for the given role.
        """
        term_alias = "s1" if role == "head" else "s2"
        other_alias = "s2" if role == "head" else "s1"
        join_col = role  # "head" or "dep"
        other_col = "dep" if role == "head" else "head"
        lc = f"AND{_lexeme_clause(lexeme, second=(role == 'dep'))}"
        return f"""
    SELECT STRAIGHT_JOIN
        '{role}' AS role,
        f.id AS relation_id,
        f.rel,
        f.head,
        s1.string AS head_string,
        s1.stringextra AS head_extra,
        s1.pos AS head_pos,
        f.dep,
        s2.string AS dep_string,
        s2.stringextra AS dep_extra,
        s2.pos AS dep_pos,
        f.freq,
        r.freq AS rel_freq,
        hr.freq AS head_rel_freq,
        dr.freq AS dep_rel_freq,
        '{corpus_label}' AS corpus
    FROM `{strings}` AS {term_alias}
    JOIN `{main}` AS f ON {term_alias}.id = f.{join_col}
    JOIN `{strings}` AS {other_alias} ON f.{other_col} = {other_alias}.id
    JOIN `{rel_table}` AS r ON f.rel = r.rel
    JOIN `{head_rel}` AS hr ON f.head = hr.head AND f.rel = hr.rel
    JOIN `{dep_rel}` AS dr ON f.dep = dr.dep AND f.rel = dr.rel
    WHERE
        {term_alias}.string = :term
        {lc}
        {freq_clause}
    """

    sql = f"{_role_select('head')} UNION ALL {_role_select('dep')}"
    return sql, params


def _build_split_triples_query(
    corpus: str,
    *,
    lexeme: bool,
    min_freq: int | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> tuple[str, dict[str, int]]:
    """Build SQL query for relation triples (split tables only).

    Args:
        corpus: The corpus name.
        lexeme: If `True`, select lexemes; if `False`, select word forms.
        min_freq: Minimum frequency filter or `None`.
        start_year: Start year (inclusive) or `None`.
        end_year: End year (inclusive) or `None`.

    Returns:
        A tuple containing the SQL query as a string and a dictionary of parameters.
    """
    tables = _table_names(corpus, split=True)
    lexeme_clause_1 = _lexeme_clause(lexeme)
    lexeme_clause_2 = _lexeme_clause(lexeme, second=True)
    freq_clause = ""
    params: dict[str, int] = {}
    if min_freq is not None:
        freq_clause = " AND f.freq >= :min_freq"
        params["min_freq"] = min_freq
    year_clause, year_params = _year_clause("f.yearfrom", start_year, end_year, prefix="triple")
    params.update(year_params)
    if year_clause:
        year_clause = "AND " + year_clause
    corpus_label = utils.sql_escape(corpus.upper())
    sql = f"""
    WITH target AS (
        SELECT s.id
        FROM `{tables["strings"]}` AS s
        WHERE s.string = :term
    )
    SELECT
        'head' AS role,
        f.id AS relation_id,
        f.yearfrom,
        f.rel,
        f.head,
        hs.string AS head_string,
        hs.stringextra AS head_extra,
        hs.pos AS head_pos,
        f.dep,
        ds.string AS dep_string,
        ds.pos AS dep_pos,
        ds.stringextra AS dep_extra,
        f.freq,
        '{corpus_label}' AS corpus
    FROM `{tables["main"]}` AS f
    JOIN target AS t ON f.head = t.id
    JOIN `{tables["strings"]}` AS hs ON f.head = hs.id
    JOIN `{tables["strings"]}` AS ds ON f.dep = ds.id
    WHERE
    {lexeme_clause_1}
    {year_clause}
    {freq_clause}
    UNION ALL
    SELECT
        'dep' AS role,
        f.id AS relation_id,
        f.yearfrom,
        f.rel,
        f.head,
        hs.string AS head_string,
        hs.stringextra AS head_extra,
        hs.pos AS head_pos,
        f.dep,
        ds.string AS dep_string,
        ds.pos AS dep_pos,
        ds.stringextra AS dep_extra,
        f.freq,
        '{corpus_label}' AS corpus
    FROM `{tables["main"]}` AS f
    JOIN target AS t ON f.dep = t.id
    JOIN `{tables["strings"]}` AS hs ON f.head = hs.id
    JOIN `{tables["strings"]}` AS ds ON f.dep = ds.id
    WHERE
    {lexeme_clause_2}
    {year_clause}
    {freq_clause}
    ORDER BY yearfrom, role, freq DESC
    """
    return sql, params


def _build_head_or_dep_query(
    corpus: str,
    *,
    role: Literal["head", "dep"],
    lexeme: bool,
    start_year: int | None = None,
    end_year: int | None = None,
) -> tuple[str, dict[str, int]]:
    """Build SQL query for head or dep relations (split tables only).

    Args:
        corpus: The corpus name.
        role: Whether to build a query for "head" or "dep" relations.
        lexeme: If `True`, select lexemes; if `False`, select word forms.
        start_year: Start year (inclusive) or `None`.
        end_year: End year (inclusive) or `None`.

    Returns:
        A tuple containing the SQL query as a string and a dictionary of parameters.
    """
    tables = _table_names(corpus, split=True)
    lexeme_clause_1 = _lexeme_clause(lexeme)
    lexeme_clause_2 = _lexeme_clause(lexeme, second=True)
    scope_clause, scope_params = _year_clause("f.yearfrom", start_year, end_year, prefix="scope")
    if scope_clause:
        scope_clause = "AND " + scope_clause

    # Role-specific aliases and table/column names
    alias = "hr" if role == "head" else "dr"
    col = role  # "head" or "dep"
    str_alias = "hs" if role == "head" else "ds"
    freq_alias = f"{col}_rel_freq"
    rel_table_key = f"{col}_rel"

    year_clause, year_params = _year_clause(f"{alias}.yearfrom", start_year, end_year, prefix=role)
    sql = f"""
    WITH target AS (
        SELECT s.id
        FROM `{tables["strings"]}` AS s
        WHERE s.string = :term
    ),
    {col}_scope AS (
        SELECT DISTINCT f.{col}, f.rel
        FROM `{tables["main"]}` AS f
        JOIN target AS t ON f.head = t.id
        WHERE
        {lexeme_clause_1}
        {scope_clause}
        UNION
        SELECT DISTINCT f.{col}, f.rel
        FROM `{tables["main"]}` AS f
        JOIN target AS t ON f.dep = t.id
        WHERE
        {lexeme_clause_2}
        {scope_clause}
    )
    SELECT
        {alias}.{col},
        {str_alias}.string AS {col}_string,
        {str_alias}.stringextra AS {col}_extra,
        {str_alias}.pos AS {col}_pos,
        {alias}.rel,
        {alias}.yearfrom,
        {alias}.freq AS {freq_alias},
        rr.freq AS rel_freq
    FROM {col}_scope AS sco
    JOIN `{tables[rel_table_key]}` AS {alias}
    ON {alias}.{col} = sco.{col} AND {alias}.rel = sco.rel
    JOIN `{tables["rel"]}` AS rr
    ON rr.rel = {alias}.rel AND rr.yearfrom <=> {alias}.yearfrom
    JOIN `{tables["strings"]}` AS {str_alias}
    ON {alias}.{col} = {str_alias}.id
    {"WHERE " + year_clause if year_clause else ""}
    ORDER BY {alias}.{col}, {alias}.rel, {alias}.yearfrom
    """
    params: dict[str, int] = {}
    params.update(scope_params)
    params.update(year_params)
    return sql, params


def _build_rel_query(
    corpus: str,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
) -> tuple[str, dict[str, int]]:
    """Build SQL query for relation frequencies in split tables.

    Args:
        corpus: The corpus name.
        start_year: Start year (inclusive) or `None`.
        end_year: End year (inclusive) or `None`.

    Returns:
        A tuple containing the SQL query as a string and a dictionary of parameters.
    """
    tables = _table_names(corpus, split=True)
    year_clause, year_params = _year_clause("yearfrom", start_year, end_year, prefix="rel")
    sql = f"""
    SELECT rel, yearfrom, freq AS rel_freq
    FROM `{tables["rel"]}`
    {"WHERE " + year_clause if year_clause else ""}
    ORDER BY rel, yearfrom
    """
    return sql, year_params


def _compute_period_bounds(
    span_length: int,
    start_year: int | None,
    end_year: int | None,
    available_years: list[int],
    *,
    align: PeriodAlign = PeriodAlign.newest,
) -> tuple[int, int, int]:
    """Compute the period bounds for the query.

    Args:
        span_length: The length of the span in years.
        start_year: The requested start year or `None`.
        end_year: The requested end year or `None`.
        available_years: List of years for which data is available.
        align: Whether to anchor periods to the oldest or newest year.

    Returns:
        A tuple containing the computed (min_year, max_year, period_origin).

    Raises:
        ValueError: If no data is available or if the parameters are invalid.
    """
    if not available_years:
        raise ValueError("No data returned for the requested period.")
    min_year = start_year if start_year is not None else min(available_years)
    max_year = end_year if end_year is not None else max(available_years)
    if span_length < 1:
        raise ValueError("span_length must be at least 1.")
    if min_year > max_year:
        raise ValueError("start_year is greater than end_year.")
    period_origin = min_year if align is PeriodAlign.oldest else max_year - span_length + 1
    return min_year, max_year, period_origin


def _build_overall_only_relations(
    corpora: list[str],
    corpus_results: dict[str, Any],
    *,
    sort_field: str,
    max_results: int,
    corpus_size: int,
    measures: Container[Measures],
) -> list[dict[str, str | int | float]]:
    """Build aggregated relation statistics for overall (non-split) data, based on the results from multiple corpora.

    Args:
        corpora: List of corpus names.
        corpus_results: Dictionary of corpus data.
        sort_field: The field to sort by.
        max_results: Maximum number of results per relation and direction.
        corpus_size: Total size of the corpora for relative frequency calculation.
        measures: Set of measures to include in the output.

    Returns:
        List of overall relation statistics.
    """
    relation_buckets: dict[RelationKey, dict[str, Any]] = {}

    # Aggregate data across corpora
    for corpus in corpora:
        data = corpus_results.get(corpus)
        if not data:
            continue

        triples: list[dict[str, Any]] = data.get("triples", [])
        head_rel_map: Counter[tuple[int, str, str]] = data.get("head_rel_map", Counter())
        dep_rel_map: Counter[tuple[int, str, str, str]] = data.get("dep_rel_map", Counter())
        rel_map: Counter[str] = data.get("rel_map", Counter())

        for row in triples:
            head_id = row["head"]
            head_pos = row["head_pos"]
            dep_id = row["dep"]
            dep_pos = row["dep_pos"]
            dep_extra = row["dep_extra"]
            rel = row["rel"]
            freq = row["freq"]
            if freq <= 0:
                continue

            head_rel_freq = row.get("head_rel_freq", head_rel_map.get((head_id, head_pos, rel), 0))
            dep_rel_freq = row.get("dep_rel_freq", dep_rel_map.get((dep_id, dep_pos, dep_extra, rel), 0))
            rel_freq = row.get("rel_freq", rel_map.get(rel, 0))

            aggregate_key: RelationKey = (
                row["role"],
                row["head_string"],
                head_pos,
                rel,
                row["dep_string"],
                dep_pos,
                dep_extra,
            )
            bucket = relation_buckets.setdefault(
                aggregate_key,
                {
                    "role": row["role"],
                    "rel": rel,
                    "head": row["head_string"],
                    "headpos": head_pos,
                    "dep": row["dep_string"],
                    "deppos": dep_pos,
                    "depextra": dep_extra,
                    "freq": 0,
                    "head_rel_freq": 0,
                    "dep_rel_freq": 0,
                    "rel_freq": 0,
                    "source": set(),
                },
            )
            bucket["freq"] += freq
            bucket["head_rel_freq"] += head_rel_freq
            bucket["dep_rel_freq"] += dep_rel_freq
            bucket["rel_freq"] += rel_freq
            rel_id = row.get("relation_id")
            if rel_id is None:
                continue
            bucket["source"].add(f"{corpus}:{rel_id}")

    if not relation_buckets:
        return []

    relation_entries = []

    # Compute MI and build entries
    for key, bucket in relation_buckets.items():
        freq = bucket["freq"]
        head_rel_freq = bucket["head_rel_freq"]
        dep_rel_freq = bucket["dep_rel_freq"]
        rel_freq = bucket["rel_freq"]
        mi_value = _calc_mi(freq, head_rel_freq, dep_rel_freq, rel_freq)
        relation_entries.append(
            {
                "key": key,
                "role": bucket["role"],
                "rel": bucket["rel"],
                "head": bucket["head"],
                "headpos": bucket["headpos"],
                "dep": bucket["dep"],
                "deppos": bucket["deppos"],
                "depextra": bucket["depextra"],
                "freq": freq,
                "freq_relative": _calc_freq_relative(freq, corpus_size),
                "mi": mi_value,
                "rmi": _calc_rmi(mi_value, rel_freq),
                "source": sorted(bucket["source"]),
            }
        )

    # Include tie-break fields here for reproducible sorting
    relation_entries.sort(
        key=lambda entry: (entry["rel"], entry.get(sort_field, entry["mi"]), entry["role"], entry["key"]),
        reverse=True,
    )
    counters: Counter[tuple[str, str]] = Counter()
    selected_entries = []
    for entry in relation_entries:
        rel_name = entry["rel"]
        key = (rel_name, entry["role"])
        counters[key] += 1
        if max_results and counters[key] > max_results:
            continue
        selected_entries.append(entry)

    return [_relation_output(entry, measures) for entry in selected_entries]


class _WordPictureAccumulator:
    """Accumulator for word picture relation data across corpora.

    This is only used together with the split (per-year) data.
    """

    def __init__(self) -> None:
        self.triple_strings: dict[RelationKey, dict[str, str]] = {}
        self.triple_freqs: dict[tuple[RelationKey, int | None], int] = defaultdict(int)
        self.head_rel_map: dict[tuple[str, str, str, int | None], int] = defaultdict(int)
        self.dep_rel_map: dict[tuple[str, str, str, str, int | None], int] = defaultdict(int)
        self.rel_year_map: dict[tuple[str, int | None], int] = defaultdict(int)
        self.years_by_rel: dict[str, set[int | None]] = defaultdict(set)
        self.sources: dict[RelationKey, set[str]] = defaultdict(set)
        self.sources_year: dict[tuple[RelationKey, int | None], set[str]] = defaultdict(set)

    def add_corpus_rows(self, corpus: str, data: dict[str, list[dict]]) -> None:
        """Add rows from a corpus to the accumulator."""
        triples = data.get("triples", [])
        for row in triples:
            year = row["yearfrom"]
            role = row["role"]
            head = row["head_string"]
            head_pos = row["head_pos"]
            dep = row["dep_string"]
            dep_pos = row["dep_pos"]
            dep_extra = row["dep_extra"]
            rel = row["rel"]
            freq = row["freq"]
            key: RelationKey = (role, head, head_pos, rel, dep, dep_pos, dep_extra)
            if key not in self.triple_strings:
                self.triple_strings[key] = {
                    "head_string": head,
                    "dep_string": dep,
                }
            self.triple_freqs[key, year] += freq
            self.years_by_rel[rel].add(year)
            relation_id = row["relation_id"]
            source_id = f"{corpus}:{relation_id}"
            self.sources[key].add(source_id)
            self.sources_year[key, year].add(source_id)

        heads = data.get("heads", [])
        for row in heads:
            year = row["yearfrom"]
            head = row["head_string"]
            head_pos = row["head_pos"]
            rel = row["rel"]
            freq = row["head_rel_freq"]
            self.head_rel_map[head, head_pos, rel, year] += freq

        deps = data.get("deps", [])
        for row in deps:
            year = row["yearfrom"]
            dep = row["dep_string"]
            dep_pos = row["dep_pos"]
            dep_extra = row["dep_extra"]
            rel = row["rel"]
            freq = row["dep_rel_freq"]
            self.dep_rel_map[dep, dep_pos, dep_extra, rel, year] += freq

        rels = data.get("rels", [])
        for row in rels:
            year = row["yearfrom"]
            rel = row["rel"]
            freq = row["rel_freq"]
            self.rel_year_map[rel, year] += freq
            self.years_by_rel[rel].add(year)

    def _get_year_range(self) -> tuple[list[int], bool]:
        """Get the range of years from accumulated data.

        Returns:
            Tuple of (sorted list of dated years, whether undated data exists).
        """
        years_set = {year for rel_years in self.years_by_rel.values() for year in rel_years}
        has_undated = None in years_set
        dated_years = sorted(year for year in years_set if year is not None)
        return dated_years, has_undated

    def _get_years_for_relation(self, rel: str) -> Sequence[int | None]:
        """Get sorted years for a specific relation, with None at the end if present.

        Args:
            rel: The relation name.

        Returns:
            List of years (int or None) for this relation.
        """
        rel_years_raw = self.years_by_rel.get(rel, set())
        rel_years = sorted(year for year in rel_years_raw if year is not None)
        if None in rel_years_raw:
            return [*rel_years, None]
        return rel_years

    def _get_freqs_for_year(self, key: RelationKey, year: int | None) -> tuple[int, int, int, int]:
        """Get frequency values for a specific key and year.

        Args:
            key: The relation key tuple.
            year: The year (or None for undated).

        Returns:
            Tuple of (freq, head_rel_freq, dep_rel_freq, rel_freq).
        """
        _, head, head_pos, rel, dep, dep_pos, dep_extra = key
        freq = self.triple_freqs.get((key, year), 0)
        head_rel_freq = self.head_rel_map.get((head, head_pos, rel, year), 0)
        dep_rel_freq = self.dep_rel_map.get((dep, dep_pos, dep_extra, rel, year), 0)
        rel_freq = self.rel_year_map.get((rel, year), 0)
        return freq, head_rel_freq, dep_rel_freq, rel_freq

    def _build_year_entry(
        self,
        key: RelationKey,
        year: int | None,
        freq: int,
        head_rel_freq: int,
        dep_rel_freq: int,
        rel_freq: int,
        mi_value: float,
        corpus_size_per_year: dict[int | None, int],
    ) -> dict[str, object]:
        """Build a per-year entry dictionary.

        Args:
            key: The relation key tuple.
            year: The year.
            freq: The frequency.
            head_rel_freq: Head-relation frequency.
            dep_rel_freq: Dep-relation frequency.
            rel_freq: Relation frequency.
            mi_value: The MI value.
            corpus_size_per_year: Mapping of year to corpus size.

        Returns:
            Dictionary with year entry data.
        """
        year_corpus_size = corpus_size_per_year.get(year, 0)
        return {
            "period_start": year,
            "period_end": year,
            "freq": freq,
            "freq_relative": _calc_freq_relative(freq, year_corpus_size),
            "head_rel_freq": head_rel_freq,
            "dep_rel_freq": dep_rel_freq,
            "rel_freq": rel_freq,
            "mi": mi_value,
            "rmi": _calc_rmi(mi_value, rel_freq),
            "source": sorted(self.sources_year.get((key, year), set())),
        }

    def _accumulate_period(
        self,
        period_accumulator: dict[tuple[RelationKey, int | None], dict],
        key: RelationKey,
        year: int | None,
        freq: int,
        head_rel_freq: int,
        dep_rel_freq: int,
        rel_freq: int,
        period_origin: int | None,
        period_min: int | None,
        period_max: int | None,
        span_length: int,
        corpus_size_per_year: dict[int | None, int],
    ) -> None:
        """Accumulate data into a period bucket.

        Args:
            period_accumulator: The period accumulator dictionary.
            key: The relation key tuple.
            year: The year.
            freq: The frequency.
            head_rel_freq: Head-relation frequency.
            dep_rel_freq: Dep-relation frequency.
            rel_freq: Relation frequency.
            period_origin: The origin year for period calculation.
            period_min: Minimum period year.
            period_max: Maximum period year.
            span_length: Length of each period.
            corpus_size_per_year: Mapping of year to corpus size.
        """
        if year is None or period_origin is None or period_min is None or period_max is None:
            period_index = None
            period_start = None
            period_end = None
        else:
            period_index = (year - period_origin) // span_length
            period_start = period_origin + period_index * span_length
            period_end = min(period_start + span_length - 1, period_max)
            period_start = max(period_start, period_min)

        bucket_key = (key, period_index)
        bucket = period_accumulator.get(bucket_key)
        if bucket is None:
            bucket = {
                "period_start": period_start,
                "period_end": period_end,
                "freq_sum": 0,
                "head_rel_sum": 0,
                "dep_rel_sum": 0,
                "rel_sum": 0,
                "corpus_size_sum": 0,
                "sources": set(),
            }
            period_accumulator[bucket_key] = bucket

        bucket["freq_sum"] += freq
        bucket["head_rel_sum"] += head_rel_freq
        bucket["dep_rel_sum"] += dep_rel_freq
        bucket["rel_sum"] += rel_freq
        bucket["corpus_size_sum"] += corpus_size_per_year.get(year, 0)
        bucket["sources"].update(self.sources_year.get((key, year), set()))

    @staticmethod
    def _finalize_periods(
        period_accumulator: dict[tuple[RelationKey, int | None], dict],
    ) -> dict[RelationKey, list[dict]]:
        """Convert period accumulator to final per-period map.

        Args:
            period_accumulator: The accumulated period data.

        Returns:
            Mapping of relation key to list of period entries.
        """
        per_period_map: dict[RelationKey, list[dict]] = defaultdict(list)
        for (key, _), bucket in period_accumulator.items():
            freq_sum = bucket["freq_sum"]
            if freq_sum == 0:
                continue
            head_sum = bucket["head_rel_sum"]
            dep_sum = bucket["dep_rel_sum"]
            rel_sum = bucket["rel_sum"]
            corpus_size_sum = bucket["corpus_size_sum"]
            mi_value = _calc_mi(freq_sum, head_sum, dep_sum, rel_sum)
            per_period_map[key].append(
                {
                    "period_start": bucket["period_start"],
                    "period_end": bucket["period_end"],
                    "freq": freq_sum,
                    "freq_relative": _calc_freq_relative(freq_sum, corpus_size_sum),
                    "head_rel_freq": head_sum,
                    "dep_rel_freq": dep_sum,
                    "rel_freq": rel_sum,
                    "mi": mi_value,
                    "rmi": _calc_rmi(mi_value, rel_sum),
                    "source": sorted(bucket["sources"]),
                }
            )
        return per_period_map

    @staticmethod
    def _finalize_overall(overall_accumulator: dict[RelationKey, dict]) -> dict[RelationKey, dict]:
        """Convert overall accumulator to final overall map.

        Args:
            overall_accumulator: The accumulated overall data.

        Returns:
            Mapping of relation key to overall statistics.
        """
        overall_map: dict[RelationKey, dict] = {}
        for key, bucket in overall_accumulator.items():
            freq_sum = bucket["freq_sum"]
            if freq_sum == 0:
                continue
            head_sum = bucket["head_rel_sum"]
            dep_sum = bucket["dep_rel_sum"]
            rel_sum = bucket["rel_sum"]
            mi_value = _calc_mi(freq_sum, head_sum, dep_sum, rel_sum)
            overall_map[key] = {
                "freq": freq_sum,
                "head_rel_freq": head_sum,
                "dep_rel_freq": dep_sum,
                "rel_freq": rel_sum,
                "mi": mi_value,
            }
        return overall_map

    def build(
        self,
        span_length: int,
        include_years: bool,
        include_periods: bool,
        corpus_size_per_year: dict[int | None, int],
        *,
        start_year: int | None = None,
        end_year: int | None = None,
        period_align: PeriodAlign = PeriodAlign.newest,
        compute_overall: bool = True,
    ) -> tuple[
        dict[RelationKey, dict],
        dict[RelationKey, list[dict]],
        dict[RelationKey, list[dict]],
        tuple[int, int] | None,
    ]:
        """Build the final relation maps from accumulated data.

        Args:
            span_length: The period span length in years.
            include_years: Whether to include per-year data.
            include_periods: Whether to include per-period data.
            corpus_size_per_year: Mapping of year to corpus size.
            start_year: Optional start year filter.
            end_year: Optional end year filter.
            period_align: Alignment for periods ("oldest" or "newest").
            compute_overall: Whether to compute overall statistics.

        Returns:
            Tuple of (overall_map, per_year_map, per_period_map, bounds).
        """
        if not self.triple_strings:
            return {}, {}, {}, None

        dated_years, has_undated = self._get_year_range()
        if not dated_years and not has_undated:
            return {}, {}, {}, None

        # Compute period bounds
        period_min: int | None = None
        period_max: int | None = None
        period_origin: int | None = None
        if dated_years:
            period_min, period_max, period_origin = _compute_period_bounds(
                span_length, start_year, end_year, dated_years, align=period_align
            )

        # Initialize accumulators
        per_year_map: dict[RelationKey, list[dict]] = defaultdict(list) if include_years else {}
        period_accumulator: dict[tuple[RelationKey, int | None], dict] = {}
        overall_accumulator: dict[RelationKey, dict] = {}
        rel_years_map = {rel: self._get_years_for_relation(rel) for rel in self.years_by_rel}

        # Process each triple
        for key in self.triple_strings:
            rel = key[3]
            for year in rel_years_map.get(rel, []):
                freq, head_rel_freq, dep_rel_freq, rel_freq = self._get_freqs_for_year(key, year)

                # This check is probably not necessary
                if freq == 0 and head_rel_freq == 0 and dep_rel_freq == 0 and rel_freq == 0:
                    continue

                mi_value = _calc_mi(freq, head_rel_freq, dep_rel_freq, rel_freq)

                # Per-year accumulation
                if include_years and freq > 0:
                    per_year_map[key].append(
                        self._build_year_entry(
                            key, year, freq, head_rel_freq, dep_rel_freq, rel_freq, mi_value, corpus_size_per_year
                        )
                    )

                # Period accumulation
                if include_periods:
                    self._accumulate_period(
                        period_accumulator,
                        key,
                        year,
                        freq,
                        head_rel_freq,
                        dep_rel_freq,
                        rel_freq,
                        period_origin,
                        period_min,
                        period_max,
                        span_length,
                        corpus_size_per_year,
                    )

                # Overall accumulation
                if compute_overall:
                    bucket = overall_accumulator.setdefault(
                        key, {"freq_sum": 0, "head_rel_sum": 0, "dep_rel_sum": 0, "rel_sum": 0}
                    )
                    bucket["freq_sum"] += freq
                    bucket["head_rel_sum"] += head_rel_freq
                    bucket["dep_rel_sum"] += dep_rel_freq
                    bucket["rel_sum"] += rel_freq

        # Finalize results
        per_period_map = self._finalize_periods(period_accumulator) if include_periods else {}
        overall_map = self._finalize_overall(overall_accumulator) if compute_overall else {}
        bounds = (period_min, period_max) if period_min is not None and period_max is not None else None

        return overall_map, per_year_map, per_period_map, bounds


async def _fetch_mappings(
    conn: AsyncConnection,
    sql: str,
    params: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Execute SQL and return mapping rows as plain dictionaries."""  # noqa: DOC201
    result = await conn.execute(text(sql), params or {})
    return [dict(row) for row in result.mappings().all()]


async def _fetch_split_relation_rows(
    conn: AsyncConnection,
    corpus: str,
    term: str,
    is_lexeme: bool,
    min_freq: int | None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> dict:
    """Fetch split (yearly) relation rows for a specific corpus.

    Args:
        conn: Async SQLAlchemy connection.
        corpus: The corpus name.
        term: The target term.
        is_lexeme: If `True`, select lexemes; if `False`, select word forms.
        min_freq: Minimum frequency filter or `None`.
        start_year: Start year (inclusive) or `None`.
        end_year: End year (inclusive) or `None`.

    Returns:
        A dictionary containing lists of relation rows.
    """
    triple_sql, triple_params = _build_split_triples_query(
        corpus,
        lexeme=is_lexeme,
        min_freq=min_freq,
        start_year=start_year,
        end_year=end_year,
    )
    head_sql, head_params = _build_head_or_dep_query(
        corpus, role="head", lexeme=is_lexeme, start_year=start_year, end_year=end_year
    )
    dep_sql, dep_params = _build_head_or_dep_query(
        corpus, role="dep", lexeme=is_lexeme, start_year=start_year, end_year=end_year
    )
    rel_sql, rel_params = _build_rel_query(corpus, start_year=start_year, end_year=end_year)

    triples, heads, deps, rels = await asyncio.gather(
        _fetch_mappings(conn, triple_sql, {"term": term, **triple_params}),
        _fetch_mappings(conn, head_sql, {"term": term, **head_params}),
        _fetch_mappings(conn, dep_sql, {"term": term, **dep_params}),
        _fetch_mappings(conn, rel_sql, rel_params),
    )

    return {
        "triples": triples,
        "heads": heads,
        "deps": deps,
        "rels": rels,
    }


async def _fetch_overall_relation_rows(
    conn: AsyncConnection,
    corpus: str,
    term: str,
    is_lexeme: bool,
    min_freq: int | None,
) -> dict[str, object]:
    """Fetch combined overall relation rows using query optimized for overall data.

    Args:
        conn: Async SQLAlchemy connection.
        corpus: The corpus name.
        term: The target term.
        is_lexeme: If `True`, select lexemes; if `False`, select word forms.
        min_freq: Minimum frequency filter or `None`.

    Returns:
        A dictionary containing relation rows and frequency maps.
    """
    sql, params = _build_overall_triples_query(corpus, lexeme=is_lexeme, min_freq=min_freq)
    triples = await _fetch_mappings(conn, sql, {"term": term, **params})

    # Maps for deduplicating head/dep/rel frequencies across rows
    head_rel_map: Counter[tuple[int, str, str]] = Counter()
    dep_rel_map: Counter[tuple[int, str, str, str]] = Counter()
    rel_map: Counter[str] = Counter()

    for row in triples:
        head_rel_map[row["head"], row["head_pos"], row["rel"]] = row["head_rel_freq"]
        dep_rel_map[row["dep"], row["dep_pos"], row["dep_extra"], row["rel"]] = row["dep_rel_freq"]
        rel_map[row["rel"]] = row["rel_freq"]

    return {
        "triples": triples,
        "head_rel_map": head_rel_map,
        "dep_rel_map": dep_rel_map,
        "rel_map": rel_map,
    }


def _build_time_rows(
    acc: "_WordPictureAccumulator",
    data_map: dict,
    keys: list[tuple] | None = None,
) -> list[dict[str, object]]:
    """Build per-year or per-period relation rows.

    Args:
        acc: The word picture accumulator.
        data_map: Mapping of keys to time-based data.
        keys: Optional list of keys to include.

    Returns:
        List of time-based relation rows.
    """
    rows: list[dict[str, object]] = []
    items = ((key, data_map.get(key, [])) for key in keys) if keys is not None else data_map.items()
    for key, entries in items:
        if not entries:
            continue
        strings = acc.triple_strings.get(key)
        if not strings:
            continue
        role, _, head_pos, rel, _, dep_pos, dep_extra = key
        rows.extend(
            {
                "role": role,
                "head": strings["head_string"],
                "headpos": head_pos,
                "rel": rel,
                "dep": strings["dep_string"],
                "deppos": dep_pos,
                "depextra": dep_extra,
                "period_start": row["period_start"],
                "period_end": row["period_end"],
                "freq": row["freq"],
                "freq_relative": row["freq_relative"],
                "head_rel_freq": row["head_rel_freq"],
                "dep_rel_freq": row["dep_rel_freq"],
                "rel_freq": row["rel_freq"],
                "mi": row["mi"],
                "rmi": row["rmi"],
                "source": row["source"],
            }
            for row in entries
        )
    return rows


def _period_bucket_key(period_start: object, period_end: object, period_size: int) -> str:
    """Return the serialized bucket key for period-based output."""
    if period_size > 1:
        return f"{period_start}-{period_end}" if period_start is not None and period_end is not None else ""
    return str(period_start if period_start is not None else "")


def _build_period_token_totals(
    corpus_size_per_year: Mapping[int | None, int],
    *,
    period_size: int,
    period_align: PeriodAlign,
    start_year: int | None,
    end_year: int | None,
    bounds: tuple[int, int] | None = None,
) -> dict[str, int]:
    """Aggregate total token frequencies into the same period buckets as `relations_time`.

    Args:
        corpus_size_per_year: Mapping of year to total token frequency for selected corpora.
        period_size: Period size in years.
        period_align: Alignment mode used for period bucket boundaries.
        start_year: Optional start year filter.
        end_year: Optional end year filter.
        bounds: Optional precomputed period bounds.

    Returns:
        Mapping from serialized period keys to token totals.
    """
    if not corpus_size_per_year:
        return {}

    period_min: int | None = None
    period_max: int | None = None
    period_origin: int | None = None

    dated_years = sorted(year for year in corpus_size_per_year if year is not None)
    if dated_years:
        if bounds is not None:
            period_min, period_max = bounds
            period_origin = period_min if period_align is PeriodAlign.oldest else period_max - period_size + 1
        else:
            period_min, period_max, period_origin = _compute_period_bounds(
                period_size,
                start_year,
                end_year,
                dated_years,
                align=period_align,
            )

    bucket_totals: dict[tuple[int | None, int | None], int] = defaultdict(int)
    for year, freq in corpus_size_per_year.items():
        period_start: int | None
        period_end: int | None

        if (
            year is not None
            and period_min is not None
            and period_max is not None
            and not (period_min <= year <= period_max)
        ):
            continue

        if year is None:
            period_start = None
            period_end = None
        elif period_size == 1:
            period_start = year
            period_end = year
        else:
            assert period_origin is not None
            assert period_min is not None
            assert period_max is not None
            period_index = (year - period_origin) // period_size
            raw_start = period_origin + period_index * period_size
            period_end = min(raw_start + period_size - 1, period_max)
            period_start = max(raw_start, period_min)

        bucket_totals[period_start, period_end] += int(freq)

    sorted_bucket_keys = sorted(
        bucket_totals,
        key=lambda key: (
            1 if key[0] is None else 0,
            key[0] if key[0] is not None else float("inf"),
            key[1] if key[1] is not None else float("inf"),
        ),
    )
    return {
        _period_bucket_key(period_start, period_end, period_size): bucket_totals[period_start, period_end]
        for period_start, period_end in sorted_bucket_keys
    }


def _limit_rows_per_bucket(
    rows: list[dict[str, object]],
    bucket_field: str,
    sort_field: str,
    max_results: int,
) -> list[dict[str, object]]:
    """Apply per-bucket result limiting.

    Args:
        rows: List of relation rows.
        bucket_field: The field to use for bucketing (e.g., "year").
        sort_field: The field to sort by within each bucket (e.g., "freq" or "mi").
        max_results: Maximum number of results per bucket and direction.

    Returns:
        List of limited relation rows.
    """
    if not rows or not max_results:
        return rows
    buckets: dict[object, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[row[bucket_field]].append(row)

    limited: list[dict[str, object]] = []

    def sort_value(r: dict[str, object]) -> float:
        val = r.get(sort_field)
        if isinstance(val, (int, float)):
            return float(val)
        backup = r.get("mi")
        return float(backup) if isinstance(backup, (int, float)) else 0.0

    def bucket_sort_key(key: object) -> tuple[int, object]:
        return (1 if key is None else 0, key if key is not None else float("-inf"))

    for bucket_key in sorted(buckets, key=bucket_sort_key):
        bucket_rows = sorted(buckets[bucket_key], key=sort_value, reverse=True)
        counters: Counter[tuple[str, str]] = Counter()
        bucket_limited: list[dict[str, object]] = []
        for row in bucket_rows:
            counter_key = (str(row["rel"]), str(row["role"]))
            counters[counter_key] += 1
            if counters[counter_key] > max_results:
                continue
            bucket_limited.append(row)
        limited.extend(bucket_limited)

    return limited


SourceParam: TypeAlias = Annotated[
    list[str],
    Query(
        description="Source ids in the format `CORPUS:ID`, repeated or comma-separated.",
        examples=[["ROMI:253662,ROMI:253663"]],
    ),
    BeforeValidator(utils.split_csv),
]


async def _existing_tables(conn: AsyncConnection, pattern: str) -> set[str]:
    rows = await _fetch_mappings(conn, "SHOW TABLES LIKE :pattern", {"pattern": pattern})
    return {str(next(iter(row.values()))) for row in rows}


async def _relations_impl(
    ctx: CtxDep,
    corpora: list[str],
    term: str,
    relation_type: RelationType,
    min_freq: int | None,
    sort_field: RelationsSort,
    max_results: int,
    include_split: bool,
    period_size: int,
    period_align: PeriodAlign,
    start_year: int | None,
    end_year: int | None,
    include_overall: bool,
    max_scope: MaxScope,
    measures: Container[Measures],
    abort_signal: AbortSignal | None = None,
) -> AsyncIterator[dict]:
    """Shared implementation for `/relations` and `/relations_time`.

    Args:
        ctx: Common dependencies.
        corpora: List of corpus names.
        term: The target term.
        relation_type: Whether the target is a word or lexeme.
        min_freq: Minimum frequency filter or `None`.
        sort_field: The field to sort results by.
        max_results: Maximum number of results per relation and direction.
        include_split: Whether to include split (per-year) data.
        period_size: The size of periods in years for split data.
        period_align: Alignment for periods ("oldest" or "newest").
        start_year: Optional start year filter for split data.
        end_year: Optional end year filter for split data.
        include_overall: Whether to include overall (non-split) data.
        max_scope: How max-result limiting is scoped ("overall" or "per_period").
        measures: Which measures to include in the output.
        abort_signal: Optional signal for aborting long-running operations.

    Yields:
        Progress updates as dictionaries with keys like "progress_corpora" or "progress_{index}",
        and finally a dictionary containing the results.
    """
    await auth.check_authorization(corpora, ctx)

    if not include_split and not include_overall:
        yield {"error": "Both split and overall results are disabled."}
        return

    is_lexeme = relation_type == RelationType.lexeme
    limit_per_period = max_scope == MaxScope.per_period
    time_filter = start_year is not None or end_year is not None
    # Use split tables whenever split output or year filtering is requested
    use_split_data = include_split or time_filter
    overall_only = not include_split

    result: dict[str, Any] = {}
    corpus_results: dict[str, dict[str, Any]] = {}
    cache_prefixes: dict[str, str] = {}
    memcached_keys: dict[str, str] = {}
    cache_checksum: str | None = None
    cached_corpora: set[str] = set()

    if ctx.common.cache:
        cache_checksum = utils.get_hash((term, is_lexeme, min_freq, use_split_data, start_year, end_year))
        cache_prefixes = await caching.cache_prefix(ctx.cache, corpora)
        for corpus in corpora:
            cache_key = f"{cache_prefixes[corpus]}:relations_{cache_checksum}"
            memcached_keys[cache_key] = corpus
        cached_data = await ctx.cache.get_many(memcached_keys.keys())
        expected_keys = (
            {"triples", "heads", "deps", "rels"}
            if use_split_data
            else {"triples", "head_rel_map", "dep_rel_map", "rel_map"}
        )
        for key, data in cached_data.items():
            corpus_name = memcached_keys.get(key)
            if not corpus_name or not isinstance(data, dict):
                continue
            if not expected_keys <= data.keys():
                continue
            corpus_results[corpus_name] = data
            cached_corpora.add(corpus_name)

    async with ctx.db.async_connection() as conn:
        await conn.execute(text("SET @@session.long_query_time = 1000;"))
        tables = await _existing_tables(conn, f"{settings.DB_WP_TABLE}_%_head_rel")
        table_suffix = f"{SPLIT_SUFFIX}_head_rel" if use_split_data else "_head_rel"

        # Filter out corpora which don't exist in database
        corpora = [c for c in corpora if f"{settings.DB_WP_TABLE}_{c.upper()}{table_suffix}" in tables]
        corpora_rest = [c for c in corpora if c not in cached_corpora]

        if not corpora:
            yield {"error": "No word picture data available for the selected corpora."}
            return

        if corpora_rest and ctx.common.incremental:
            yield {"progress_corpora": list(corpora_rest)}

        progress_index = 0

        # Fetch per-corpus rows from the chosen table family
        for corpus in corpora_rest:
            if abort_signal and abort_signal.is_set():
                return
            if use_split_data:
                data = await _fetch_split_relation_rows(
                    conn,
                    corpus,
                    term,
                    is_lexeme,
                    min_freq,
                    start_year=start_year,
                    end_year=end_year,
                )
            else:
                # Neither split output nor year filtering requested: use overall-optimized query
                data = await _fetch_overall_relation_rows(conn, corpus, term, is_lexeme, min_freq)
            corpus_results[corpus] = data
            if ctx.common.cache and cache_checksum is not None:
                cache_key = f"{cache_prefixes[corpus]}:relations_{cache_checksum}"
                try:
                    await ctx.cache.add(cache_key, data)
                except CacheError:
                    pass
            if ctx.common.incremental:
                yield {f"progress_{progress_index}": {"corpus": corpus}}
                progress_index += 1

    # Fast path: overall-only with no year filtering
    if overall_only and not use_split_data:
        if Measures.freq_relative in measures:
            # Avoid calling CWB if relative frequencies are not needed, to be able to test the endpoint without CWB
            corpus_data = await info.get_corpus_info(ctx=ctx, corpora=corpora, no_combined_cache=True)
            total_corpus_size = sum(int(corpus_data["corpora"][corpus]["info"]["Size"]) for corpus in corpora)
        else:
            total_corpus_size = 0

        result["relations"] = _build_overall_only_relations(
            corpora,
            corpus_results,
            sort_field=sort_field.value,
            max_results=max_results,
            corpus_size=total_corpus_size,
            measures=measures,
        )
        yield result
        return

    # Everything past this point uses the accumulator for split/overall data

    # Get yearly size of corpora to be able to compute relative frequencies
    corpus_timedata = await timespan.get_timespan(
        ctx,
        corpora,
        granularity=params.GranularityValues.year,
        combined=False,
        per_corpus=True,
        no_combined_cache=True,
    )

    # Sum up total frequencies per year
    corpus_size_per_year: dict[int | None, int] = defaultdict(int)
    for corpus in corpus_timedata.get("corpora", {}):
        for year, freq in corpus_timedata["corpora"][corpus].items():
            corpus_size_per_year[int(year) if year != "" else None] += freq

    if time_filter:
        filtered_sizes: dict[int | None, int] = {}
        for year, freq in corpus_size_per_year.items():
            if year is None:
                filtered_sizes[year] = freq
                continue
            if start_year is not None and year < start_year:
                continue
            if end_year is not None and year > end_year:
                continue
            filtered_sizes[year] = freq
        corpus_size_per_year = filtered_sizes

    # Total corpus size across (possibly filtered) years
    total_corpus_size = sum(corpus_size_per_year.values())

    # Aggregate split rows for overall + time-sliced outputs
    acc = _WordPictureAccumulator()
    for corpus in corpora:
        if corpus_data := corpus_results.get(corpus):
            acc.add_corpus_rows(corpus, corpus_data)

    include_years = include_split and period_size == 1
    include_periods = include_split and period_size > 1
    compute_overall = include_overall or not limit_per_period  # Overall either requested or needed for scoping
    overall_map, per_year_map, per_period_map, bounds = acc.build(
        period_size,
        include_years,
        include_periods,
        period_align=period_align,
        compute_overall=compute_overall,
        corpus_size_per_year=corpus_size_per_year,
        start_year=start_year,
        end_year=end_year,
    )
    if include_split:
        result["token_frequencies"] = _build_period_token_totals(
            corpus_size_per_year,
            period_size=period_size,
            period_align=period_align,
            start_year=start_year,
            end_year=end_year,
            bounds=bounds,
        )

    overall_relation_entries = []
    if overall_map:
        for key, bucket in overall_map.items():
            strings = acc.triple_strings.get(key)
            if not strings:
                continue
            role, _, head_pos, rel, _, dep_pos, dep_extra = key
            overall_relation_entries.append(
                {
                    "role": role,
                    "key": key,
                    "rel": rel,
                    "head": strings["head_string"],
                    "headpos": head_pos,
                    "dep": strings["dep_string"],
                    "deppos": dep_pos,
                    "depextra": dep_extra,
                    "freq": int(bucket["freq"]),
                    "freq_relative": _calc_freq_relative(int(bucket["freq"]), total_corpus_size),
                    "mi": bucket["mi"],
                    "rmi": _calc_rmi(bucket["mi"], bucket["rel_freq"]),
                    "source": sorted(acc.sources.get(key, [])),
                }
            )

    selected_entries: list[dict[str, Any]] = []
    selected_keys: list[RelationKey] = []

    if overall_relation_entries:
        # We have overall entries, meaning either overall results were requested or they are needed for scoping
        # time-sliced results.

        # Sort overall entries by relation and the chosen sort field, then apply max_results per relation and role.
        overall_relation_entries.sort(
            key=lambda entry: (entry["rel"], entry.get(sort_field.value, entry["mi"])), reverse=True
        )
        counters: Counter[tuple[object, str]] = Counter()
        for entry in overall_relation_entries:
            key = (entry["rel"], entry["role"])
            counters[key] += 1
            if max_results and counters[key] > max_results:
                continue
            selected_entries.append(entry)
        if include_overall:
            result["relations"] = [_relation_output(entry, measures) for entry in selected_entries]
        if not limit_per_period:
            # max_scope=overall: time results must be scoped to selected overall relations
            selected_keys = [entry["key"] for entry in selected_entries]
    else:
        # No overall entries available
        if include_overall:
            result["relations"] = []
        if not limit_per_period:
            # max_scope=overall was requested but there are no overall entries, so return early
            if include_split:
                result["relations_time"] = {}
            yield result
            return

    # Build per-year/per-period outputs when requested
    if include_split:
        per_period_rows: list[dict[str, object]] = []
        if include_years:
            per_period_rows.extend(_build_time_rows(acc, per_year_map, None if limit_per_period else selected_keys))
        if include_periods:
            per_period_rows.extend(_build_time_rows(acc, per_period_map, None if limit_per_period else selected_keys))

        if per_period_rows:
            if limit_per_period and max_results:
                per_period_rows = _limit_rows_per_bucket(per_period_rows, "period_start", sort_field.value, max_results)
            per_period_rows.sort(
                key=lambda row: (
                    row["rel"],
                    row["head"],
                    row["dep"],
                    1 if row["period_start"] is None else 0,
                    row["period_start"],
                )
            )
            if bounds is not None:
                result["range"] = {"start": bounds[0], "end": bounds[1]}
            result["period_size"] = period_size
            grouped_time_result = {}
            for row in per_period_rows:
                bucket_key = _period_bucket_key(row["period_start"], row["period_end"], period_size)
                grouped_time_result.setdefault(bucket_key, []).append(_relation_output(row, measures))
            result["relations_time"] = grouped_time_result
        else:
            result["relations_time"] = {}

    yield result


@router.get(
    "/relations",
    response_model=None,
    responses=docs_response(RelationsResponse),
    summary="Word Picture",
    description=RELATIONS_DESCRIPTION,
)
@router.post("/relations", response_model=None, include_in_schema=False)
@api_handler
async def relations(
    ctx: CtxDep,
    corpus: params.CorpusParam,
    term: TermParam,
    relation_type: RelationTypeParam = RelationType.word,
    min_freq: MinFreqParam = None,
    max_results: MaxResultsParam = 15,
    sort: RelationsSortParam = RelationsSort.mi,
    split: RelationsSplitParam = False,
    period_size: PeriodSizeParam = 1,
    period_align: PeriodAlignParam = PeriodAlign.newest,
    start_year: YearParam = None,
    end_year: YearParam = None,
    overall: OverallParam = True,
    max_scope: MaxScopeParam = MaxScope.per_period,
    measures: MeasuresParam = tuple(Measures),
    abort_signal: AbortDep = None,
) -> AsyncIterator[dict]:
    """Calculate word picture data.

    Yields:
        Word picture relation data.
    """
    async for item in _relations_impl(
        ctx=ctx,
        corpora=corpus,
        term=term,
        relation_type=relation_type,
        min_freq=min_freq,
        sort_field=sort,
        max_results=max_results,
        include_split=split,
        period_size=period_size,
        period_align=period_align,
        start_year=start_year,
        end_year=end_year,
        include_overall=overall,
        max_scope=max_scope,
        measures=measures,
        abort_signal=abort_signal,
    ):
        yield item


@router.get(
    "/relations_time",
    response_model=None,
    responses=docs_response(RelationsResponse),
    summary="Word Picture Over Time",
    description=RELATIONS_TIME_DESCRIPTION,
)
@router.post("/relations_time", response_model=None, include_in_schema=False)
@api_handler
async def relations_time(
    ctx: CtxDep,
    corpus: params.CorpusParam,
    term: TermParam,
    relation_type: RelationTypeParam = RelationType.word,
    min_freq: MinFreqParam = None,
    max_results: MaxResultsParam = 15,
    sort: RelationsSortParam = RelationsSort.mi,
    period_size: PeriodSizeParam = 1,
    period_align: PeriodAlignParam = PeriodAlign.newest,
    start_year: YearParam = None,
    end_year: YearParam = None,
    overall: OverallParam = False,
    max_scope: MaxScopeParam = MaxScope.per_period,
    measures: MeasuresParam = tuple(Measures),
    abort_signal: AbortDep = None,
) -> AsyncIterator[dict]:
    """Calculate word picture data with time splits.

    Yields:
        Word picture relation data with time splits.
    """
    async for item in _relations_impl(
        ctx=ctx,
        corpora=corpus,
        term=term,
        relation_type=relation_type,
        min_freq=min_freq,
        sort_field=sort,
        max_results=max_results,
        include_split=True,
        period_size=period_size,
        period_align=period_align,
        start_year=start_year,
        end_year=end_year,
        include_overall=overall,
        max_scope=max_scope,
        measures=measures,
        abort_signal=abort_signal,
    ):
        yield item


def _parse_source(source: list[str]) -> dict[str, set[int]]:
    parsed: dict[str, set[int]] = defaultdict(set)
    for item in source:
        try:
            corpus, relation_id = item.split(":", 1)
            parsed[corpus.upper()].add(int(relation_id))
        except ValueError as exc:
            raise ValueError("Malformed value for key 'source'. Expected 'CORPUS:ID'.") from exc
    return parsed


async def _relations_sentences_impl(
    ctx: CtxDep,
    source: list[str],
    offset: int,
    limit: int,
    show: str,
    show_struct: str,
    default_context: str,
    yearly: bool,
    abort_signal: AbortSignal | None = None,
) -> dict[str, Any]:
    source_map = _parse_source(source)
    await auth.check_authorization(source_map.keys(), ctx)

    table_suffix = f"{SPLIT_SUFFIX}_sentences" if yearly else "_sentences"
    shown = show or "word"
    shown_structs = set(utils.split_csv(show_struct))
    debug: dict[str, Any] = {}

    sql_query_start_time = time.perf_counter()
    async with ctx.db.async_connection() as conn:
        await conn.execute(text("SET @@session.long_query_time = 1000;"))
        tables = await _existing_tables(conn, f"{settings.DB_WP_TABLE}_%{table_suffix}")
        filtered_source = sorted(
            [
                (corpus, ids)
                for corpus, ids in source_map.items()
                if f"{settings.DB_WP_TABLE}_{corpus.upper()}{table_suffix}" in tables
            ]
        )
        if not filtered_source:
            return {}
        corpora = [corpus for corpus, _ in filtered_source]

        selects: list[str] = []
        counts: list[str] = []
        for corpus, ids in filtered_source:
            ids_list = "(" + ", ".join(f"{i:d}" for i in sorted(ids)) + ")"
            corpus_table_sentences = f"{settings.DB_WP_TABLE}_{corpus.upper()}{table_suffix}"
            selects.append(
                f"""(
                    SELECT
                        S.sentence,
                        S.start,
                        S.end,
                        '{utils.sql_escape(corpus.upper())}' AS corpus
                    FROM
                        `{corpus_table_sentences}` as S
                    WHERE
                        S.id IN {ids_list}
                )"""
            )
            counts.append(
                f"""(
                    SELECT
                        '{utils.sql_escape(corpus.upper())}' AS corpus,
                        COUNT(*) AS freq
                    FROM
                        `{corpus_table_sentences}` as S
                    WHERE
                        S.id IN {ids_list}
                )"""
            )

        count_rows = await _fetch_mappings(conn, " UNION ALL ".join(counts))
        corpus_hits = {row["corpus"]: int(row["freq"]) for row in count_rows}
        sentence_rows = await _fetch_mappings(conn, " UNION ALL ".join(selects) + f" LIMIT {offset}, {limit}")
        if ctx.common.debug:
            debug["sql_time"] = time.perf_counter() - sql_query_start_time

    corpora_dict: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for row in sentence_rows:
        corpora_dict.setdefault(row["corpus"], {}).setdefault(row["sentence"], []).append((row["start"], row["end"]))

    total_hits = sum(corpus_hits.values())
    if not corpora_dict:
        return {"hits": 0}

    cqp_query_start_time = time.perf_counter()
    result: dict[str, Any] = {}
    for corpus, sids in sorted(corpora_dict.items(), key=operator.itemgetter(0)):
        if abort_signal and abort_signal.is_set():
            return result
        cqp = '<sentence_id="{}"> []* </sentence_id> within sentence'.format("|".join(set(sids.keys())))
        query_params = await query.parse_parameters(
            ctx=ctx,
            corpus=[corpus],
            cqp_query=[cqp],
            offset=0,
            limit=limit,
            show=utils.split_csv(shown),
            show_struct=["sentence_id", *shown_structs],
            default_context=default_context,
        )
        result_temp = await utils.async_generator_to_dict(
            query.perform_query(query_params, ctx, abort_signal=abort_signal)
        )

        # Loop backwards since we might be adding new items
        for i in range(len(result_temp["kwic"]) - 1, -1, -1):
            sentence = result_temp["kwic"][i]
            sid = sentence["structs"]["sentence_id"]
            relation_positions = sids[sid][0]
            sentence_start = sentence["match"]["start"]
            sentence["match"]["start"] = sentence_start + min(map(int, relation_positions)) - 1
            sentence["match"]["end"] = sentence_start + max(map(int, relation_positions))

            # If the same relation appears more than once in the same sentence,
            # append copies of the sentence as separate results
            for relation_positions in sids[sid][1:]:
                copy_sentence = deepcopy(sentence)
                copy_sentence["match"]["start"] = sentence_start + min(map(int, relation_positions)) - 1
                copy_sentence["match"]["end"] = sentence_start + max(map(int, relation_positions))
                result_temp["kwic"].insert(i + 1, copy_sentence)

        result.setdefault("kwic", []).extend(result_temp["kwic"])

    result["hits"] = total_hits
    result["corpus_hits"] = corpus_hits
    result["corpus_order"] = corpora
    if ctx.common.debug:
        debug["cqp_time"] = time.perf_counter() - cqp_query_start_time
        result["debug"] = debug
    return result


@router.get(
    "/relations_sentences",
    response_model=None,
    responses=docs_response(RelationsSentencesResponse),
    summary="Word Picture Sentences",
    description=RELATIONS_SENTENCES_DESCRIPTION,
)
@router.post("/relations_sentences", response_model=None, include_in_schema=False)
@api_handler
async def relations_sentences(
    ctx: CtxDep,
    source: SourceParam,
    offset: RelationsOffsetParam = 0,
    limit: RelationsLimitParam = 10,
    show: RelationsShowParam = "word",
    show_struct: RelationsShowStructParam = "",
    default_context: RelationsDefaultContextParam = "1 sentence",
    abort_signal: AbortDep = None,
) -> AsyncIterator[dict]:
    """Find sentences containing relations from word picture source IDs.

    Args:
        ctx: Common dependencies.
        source: List of source IDs in the format `CORPUS:ID`.
        offset: Number of sentence rows to skip.
        limit: Maximum number of sentence rows to return.
        show: Comma-separated list of token fields to include in results.
        show_struct: Comma-separated list of structural attributes to include.
        default_context: Default context size for query results (e.g., "1 sentence").
        abort_signal: Optional signal for aborting long-running operations.

    Yields:
        A dictionary containing the sentences and related metadata.
    """
    yield await _relations_sentences_impl(
        ctx=ctx,
        source=source,
        offset=offset,
        limit=limit,
        show=show,
        show_struct=show_struct,
        default_context=default_context,
        yearly=False,
        abort_signal=abort_signal,
    )


@router.get(
    "/relations_time_sentences",
    response_model=None,
    responses=docs_response(RelationsSentencesResponse),
    summary="Word Picture Time Sentences",
    description=RELATIONS_TIME_SENTENCES_DESCRIPTION,
)
@router.post("/relations_time_sentences", response_model=None, include_in_schema=False)
@api_handler
async def relations_time_sentences(
    ctx: CtxDep,
    source: SourceParam,
    offset: RelationsOffsetParam = 0,
    limit: RelationsLimitParam = 10,
    show: RelationsShowParam = "word",
    show_struct: RelationsShowStructParam = "",
    default_context: RelationsDefaultContextParam = "1 sentence",
    abort_signal: AbortDep = None,
) -> AsyncIterator[dict]:
    """Find time-split sentences containing relations from word picture source IDs.

    Args:
        ctx: Common dependencies.
        source: List of source IDs in the format `CORPUS:ID`.
        offset: Number of sentence rows to skip.
        limit: Maximum number of sentence rows to return.
        show: Comma-separated list of token fields to include in results.
        show_struct: Comma-separated list of structural attributes to include.
        default_context: Default context size for query results (e.g., "1 sentence").
        abort_signal: Optional signal for aborting long-running operations.

    Yields:
        A dictionary containing the sentences and related metadata.
    """
    yield await _relations_sentences_impl(
        ctx=ctx,
        source=source,
        offset=offset,
        limit=limit,
        show=show,
        show_struct=show_struct,
        default_context=default_context,
        yearly=True,
        abort_signal=abort_signal,
    )

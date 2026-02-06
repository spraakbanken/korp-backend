"""Views for Word Picture queries."""

import math
import operator
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from copy import deepcopy
from typing import Any, Literal

import gevent.event
from flask import Blueprint
from flask import current_app as app
from pymemcache.exceptions import MemcacheError

from korp import utils
from korp.db import mysql
from korp.memcached import memcached

from . import query, timespan

bp = Blueprint("relations", __name__)


# (role, head_id, head_pos, rel, dep_id, dep_pos, dep_extra)
RelationKey = tuple[str, int, str, str, int, str, str]


RMI_MULTIPLIER = 1_000
FREQ_RELATIVE_MULTIPLIER = 1_000_000
SPLIT_SUFFIX = "_yearly"


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


def _relation_output(entry: dict[str, object]) -> dict[str, object]:
    """Build the standard relation output dictionary.

    Args:
        entry: Dictionary containing relation data.

    Returns:
        Dictionary with the standard output fields.
    """
    return {
        "head": entry["head"],
        "headpos": entry["headpos"],
        "rel": entry["rel"],
        "dep": entry["dep"],
        "deppos": entry["deppos"],
        "depextra": entry["depextra"],
        "freq": entry["freq"],
        "freq_relative": entry["freq_relative"],
        "mi": entry["mi"],
        "rmi": entry["rmi"],
        "source": entry["source"],
    }


def _table_names(corpus: str, *, split: bool) -> dict[str, str]:
    """Return the table names for a corpus.

    Args:
        corpus: Corpus identifier.
        split: If `True`, return the per-year (split) table names; otherwise overall tables.
    """
    prefix = f"{app.config['DBWPTABLE']}_{corpus.upper()}"
    main_prefix = f"{prefix}{SPLIT_SUFFIX}" if split else prefix
    return {
        "strings": f"{prefix}_strings",  # Shared between split and overall
        "main": main_prefix,
        "head_rel": f"{main_prefix}_head_rel",
        "dep_rel": f"{main_prefix}_dep_rel",
        "rel": f"{main_prefix}_rel",
    }


def _lemgram_clause(lemgram: bool, *, second: bool = False) -> str:
    """Return SQL clause for lemgram or wordform selection.

    Args:
        lemgram: If `True`, select lemgrams; if `False`, select wordforms.
        second: If `True`, generate clause for the second query (dep); if `False`, for the first (head).

    Returns:
        SQL clause as a string.
    """
    if lemgram:
        return " f.bfhead = 1 AND f.bfdep = 1"
    if second:
        return " f.wfdep = 1"
    return " f.wfhead = 1"


def _year_clause(column: str, start_year: int | None, end_year: int | None) -> tuple[str, list[int]]:
    """Return SQL clause for year range selection.

    Args:
        column: The column name to apply the year constraints to.
        start_year: The start year (inclusive) or `None`.
        end_year: The end year (inclusive) or `None`.

    Returns:
        A tuple containing the SQL clause as a string and a list of parameters.
    """
    clauses: list[str] = []
    params: list[int] = []
    if start_year is not None:
        clauses.append(f"{column} >= %s")
        params.append(start_year)
    if end_year is not None:
        clauses.append(f"{column} <= %s")
        params.append(end_year)
    if not clauses:
        return "", params
    combined = " AND ".join(clauses)
    return f"({column} IS NULL OR ({combined}))", params


def _build_overall_triples_query(
    corpus: str,
    *,
    lemgram: bool,
    min_freq: int | None = None,
) -> tuple[str, list[object]]:
    """Build combined SQL query for overall relations (no year splitting).

    Args:
        corpus: The corpus name.
        lemgram: If `True`, select lemgrams; if `False`, select wordforms.
        min_freq: Minimum frequency filter or `None` (no minimum).

    Returns:
        A tuple containing the SQL query as a string and a list of parameters.
    """
    tables = _table_names(corpus, split=False)
    strings = tables["strings"]
    main = tables["main"]
    rel_table = tables["rel"]
    head_rel = tables["head_rel"]
    dep_rel = tables["dep_rel"]
    lemgram_clause_1 = f"AND{_lemgram_clause(lemgram)}"
    lemgram_clause_2 = f"AND{_lemgram_clause(lemgram, second=True)}"
    corpus_label = utils.sql_escape(corpus.upper())
    freq_clause = ""
    freq_params: list[object] = []
    if min_freq is not None:
        freq_clause = " AND f.freq >= %s"
        freq_params.append(min_freq)

    head_select = f"""
    SELECT STRAIGHT_JOIN
        'head' AS role,
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
    FROM `{strings}` AS s1
    JOIN `{main}` AS f ON s1.id = f.head
    JOIN `{strings}` AS s2 ON f.dep = s2.id
    JOIN `{rel_table}` AS r ON f.rel = r.rel
    JOIN `{head_rel}` AS hr ON f.head = hr.head AND f.rel = hr.rel
    JOIN `{dep_rel}` AS dr ON f.dep = dr.dep AND f.rel = dr.rel
    WHERE
        s1.string = %s
        {lemgram_clause_1}
        {freq_clause}
    """

    dep_select = f"""
    SELECT STRAIGHT_JOIN
        'dep' AS role,
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
    FROM `{strings}` AS s2
    JOIN `{main}` AS f ON s2.id = f.dep
    JOIN `{strings}` AS s1 ON f.head = s1.id
    JOIN `{rel_table}` AS r ON f.rel = r.rel
    JOIN `{head_rel}` AS hr ON f.head = hr.head AND f.rel = hr.rel
    JOIN `{dep_rel}` AS dr ON f.dep = dr.dep AND f.rel = dr.rel
    WHERE
        s2.string = %s
        {lemgram_clause_2}
        {freq_clause}
    """

    sql = f"{head_select} UNION ALL {dep_select}"
    return sql, freq_params


def _build_split_triples_query(
    corpus: str,
    *,
    lemgram: bool,
    min_freq: int | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> tuple[str, list[int]]:
    """Build SQL query for relation triples (split tables only).

    Args:
        corpus: The corpus name.
        lemgram: If `True`, select lemgrams; if `False`, select wordforms.
        min_freq: Minimum frequency filter or `None`.
        start_year: Start year (inclusive) or `None`.
        end_year: End year (inclusive) or `None`.

    Returns:
        A tuple containing the SQL query as a string and a list of parameters.
    """
    tables = _table_names(corpus, split=True)
    lemgram_clause_1 = _lemgram_clause(lemgram)
    lemgram_clause_2 = _lemgram_clause(lemgram, second=True)
    freq_clause = ""
    freq_params: list[int] = []
    if min_freq is not None:
        freq_clause = " AND f.freq >= %s"
        freq_params = [min_freq]
    year_clause, year_params = _year_clause("f.yearfrom", start_year, end_year)
    if year_clause:
        year_clause = "AND " + year_clause
    select_params = year_params + freq_params
    params = select_params + select_params
    corpus_label = utils.sql_escape(corpus.upper())
    sql = f"""
    WITH target AS (
        SELECT s.id
        FROM `{tables["strings"]}` AS s
        WHERE s.string = %s
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
    {lemgram_clause_1}
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
    {lemgram_clause_2}
    {year_clause}
    {freq_clause}
    ORDER BY yearfrom, role, freq DESC
    """
    return sql, params


def _build_head_query(
    corpus: str,
    *,
    lemgram: bool,
    start_year: int | None = None,
    end_year: int | None = None,
) -> tuple[str, list[int]]:
    """Build SQL query for head relations (split tables only).

    Args:
        corpus: The corpus name.
        lemgram: If `True`, select lemgrams; if `False`, select wordforms.
        start_year: Start year (inclusive) or `None`.
        end_year: End year (inclusive) or `None`.

    Returns:
        A tuple containing the SQL query as a string and a list of parameters.
    """
    tables = _table_names(corpus, split=True)
    lemgram_clause_1 = _lemgram_clause(lemgram)
    lemgram_clause_2 = _lemgram_clause(lemgram, second=True)
    scope_clause, scope_params = _year_clause("f.yearfrom", start_year, end_year)
    if scope_clause:
        scope_clause = "AND " + scope_clause
    year_clause, year_params = _year_clause("hr.yearfrom", start_year, end_year)
    sql = f"""
    WITH target AS (
        SELECT s.id
        FROM `{tables["strings"]}` AS s
        WHERE s.string = %s
    ),
    head_scope AS (
        SELECT DISTINCT f.head, f.rel
        FROM `{tables["main"]}` AS f
        JOIN target AS t ON f.head = t.id
        WHERE
        {lemgram_clause_1}
        {scope_clause}
        UNION
        SELECT DISTINCT f.head, f.rel
        FROM `{tables["main"]}` AS f
        JOIN target AS t ON f.dep = t.id
        WHERE
        {lemgram_clause_2}
        {scope_clause}
    )
    SELECT
        hr.head,
        hs.string AS head_string,
        hs.stringextra AS head_extra,
        hs.pos AS head_pos,
        hr.rel,
        hr.yearfrom,
        hr.freq AS head_rel_freq,
        rr.freq AS rel_freq
    FROM head_scope AS hsco
    JOIN `{tables["head_rel"]}` AS hr
    ON hr.head = hsco.head AND hr.rel = hsco.rel
    JOIN `{tables["rel"]}` AS rr
    ON rr.rel = hr.rel AND rr.yearfrom <=> hr.yearfrom
    JOIN `{tables["strings"]}` AS hs
    ON hr.head = hs.id
    {"WHERE " + year_clause if year_clause else ""}
    ORDER BY hr.head, hr.rel, hr.yearfrom
    """
    params: list[int] = scope_params + scope_params + year_params
    return sql, params


def _build_dep_query(
    corpus: str,
    *,
    lemgram: bool,
    start_year: int | None = None,
    end_year: int | None = None,
) -> tuple[str, list[int]]:
    """Build SQL query for dependent relations (split tables only).

    Args:
        corpus: The corpus name.
        lemgram: If `True`, select lemgrams; if `False`, select wordforms.
        start_year: Start year (inclusive) or `None`.
        end_year: End year (inclusive) or `None`.

    Returns:
        A tuple containing the SQL query as a string and a list of parameters.
    """
    tables = _table_names(corpus, split=True)
    lemgram_clause_1 = _lemgram_clause(lemgram)
    lemgram_clause_2 = _lemgram_clause(lemgram, second=True)
    scope_clause, scope_params = _year_clause("f.yearfrom", start_year, end_year)
    if scope_clause:
        scope_clause = "AND " + scope_clause
    year_clause, year_params = _year_clause("dr.yearfrom", start_year, end_year)
    sql = f"""
    WITH target AS (
        SELECT s.id
        FROM `{tables["strings"]}` AS s
        WHERE s.string = %s
    ),
    dep_scope AS (
        SELECT DISTINCT f.dep, f.rel
        FROM `{tables["main"]}` AS f
        JOIN target AS t ON f.head = t.id
        WHERE
        {lemgram_clause_1}
        {scope_clause}
        UNION
        SELECT DISTINCT f.dep, f.rel
        FROM `{tables["main"]}` AS f
        JOIN target AS t ON f.dep = t.id
        WHERE
        {lemgram_clause_2}
        {scope_clause}
    )
    SELECT
        dr.dep,
        ds.string AS dep_string,
        ds.pos AS dep_pos,
        ds.stringextra AS dep_extra,
        dr.rel,
        dr.yearfrom,
        dr.freq AS dep_rel_freq,
        rr.freq AS rel_freq
    FROM dep_scope AS dsco
    JOIN `{tables["dep_rel"]}` AS dr
    ON dr.dep = dsco.dep AND dr.rel = dsco.rel
    JOIN `{tables["rel"]}` AS rr
    ON rr.rel = dr.rel AND rr.yearfrom <=> dr.yearfrom
    JOIN `{tables["strings"]}` AS ds
    ON dr.dep = ds.id
    {"WHERE " + year_clause if year_clause else ""}
    ORDER BY dr.dep, dr.rel, dr.yearfrom
    """
    params: list[int] = scope_params + scope_params + year_params
    return sql, params


def _build_rel_query(
    corpus: str,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
) -> tuple[str, list[int]]:
    """Build SQL query for relation frequencies in split tables.

    Args:
        corpus: The corpus name.
        start_year: Start year (inclusive) or `None`.
        end_year: End year (inclusive) or `None`.

    Returns:
        A tuple containing the SQL query as a string and a list of parameters.
    """
    tables = _table_names(corpus, split=True)
    year_clause, year_params = _year_clause("yearfrom", start_year, end_year)
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
    align: Literal["oldest", "newest"] = "newest",
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
    period_origin = min_year if align == "oldest" else max_year - span_length + 1
    return min_year, max_year, period_origin


def _build_overall_only_relations(
    corpora: list[str],
    corpus_results: dict[str, Any],
    *,
    sort_field: str,
    max_results: int,
    corpus_size: int,
) -> list[dict[str, str | int | float | None]]:
    """Build aggregated relation statistics for overall (non-split) data, based on the results from multiple corpora.

    Args:
        corpora: List of corpus names.
        corpus_results: Dictionary of corpus data.
        sort_field: The field to sort by.
        max_results: Maximum number of results per relation and direction.
        corpus_size: Total size of the corpora for relative frequency calculation.

    Returns:
        List of overall relation statistics.
    """
    triples: list[dict[str, object]] = []
    head_rel_map: Counter[tuple[int, str, str]] = Counter()
    dep_rel_map: Counter[tuple[int, str, str, str]] = Counter()
    rel_map: Counter[str] = Counter()
    sources: dict[tuple[int, str, str, int, str, str], set[str]] = defaultdict(set)

    # Aggregate data across corpora
    for corpus in corpora:
        data = corpus_results.get(corpus)
        if not data:
            continue
        triples.extend(data.get("triples", []))
        head_rel_map.update(data.get("head_rel_map", {}))
        dep_rel_map.update(data.get("dep_rel_map", {}))
        rel_map.update(data.get("rel_map", {}))
        for row in data.get("triples", []):
            rel_id = row.get("relation_id")
            if rel_id is None:
                continue
            key = (
                row["head"],
                str(row["head_pos"] or ""),
                row["rel"],
                row["dep"],
                str(row["dep_pos"] or ""),
                str(row["dep_extra"] or ""),
            )
            sources[key].add(f"{corpus}:{rel_id}")

    if not triples:
        return []

    relation_entries = []

    # Compute MI and build entries
    for row in triples:
        head = row["head"]
        head_pos = row["head_pos"]
        dep = row["dep"]
        dep_pos = row["dep_pos"]
        dep_extra = row["dep_extra"]
        rel = row["rel"]
        freq = row["freq"]

        if freq == 0:
            continue

        head_rel_freq = head_rel_map[(head, head_pos, rel)]
        dep_rel_freq = dep_rel_map[(dep, dep_pos, dep_extra, rel)]
        rel_freq = rel_map[rel]
        mi_value = _calc_mi(freq, head_rel_freq, dep_rel_freq, rel_freq)

        key = (head, head_pos, rel, dep, dep_pos, dep_extra)
        relation_entries.append(
            {
                "key": key,
                "role": row["role"],
                "rel": rel,
                "head": row["head_string"],
                "headpos": row["head_pos"],
                "dep": row["dep_string"],
                "deppos": row["dep_pos"],
                "depextra": row["dep_extra"],
                "freq": freq,
                "freq_relative": _calc_freq_relative(freq, corpus_size),
                "mi": mi_value,
                "rmi": _calc_rmi(mi_value, rel_freq),
                "source": sorted(sources.get(key, [])),
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

    return [_relation_output(entry) for entry in selected_entries]


class _WordPictureAccumulator:
    """Accumulator for word picture relation data across corpora.

    This is only used together with the split (per-year) data.
    """

    def __init__(self) -> None:
        self.triple_strings: dict[RelationKey, dict[str, str]] = {}
        self.triple_freqs: dict[tuple[RelationKey, int | None], int] = defaultdict(int)
        self.head_rel_map: dict[tuple[int, str, str, int | None], int] = defaultdict(int)
        self.dep_rel_map: dict[tuple[int, str, str, str, int | None], int] = defaultdict(int)
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
            head = row["head"]
            head_pos = row["head_pos"]
            dep = row["dep"]
            dep_pos = row["dep_pos"]
            dep_extra = row["dep_extra"]
            rel = row["rel"]
            freq = row["freq"]
            key: RelationKey = (role, head, head_pos, rel, dep, dep_pos, dep_extra)
            if key not in self.triple_strings:
                self.triple_strings[key] = {
                    "head_string": str(row["head_string"]),
                    "dep_string": str(row["dep_string"]),
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
            head = row["head"]
            head_pos = row["head_pos"]
            rel = row["rel"]
            freq = row["head_rel_freq"]
            self.head_rel_map[head, head_pos, rel, year] += freq

        deps = data.get("deps", [])
        for row in deps:
            year = row["yearfrom"]
            dep = row["dep"]
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

    def _get_years_for_relation(self, rel: str) -> list[int | None]:
        """Get sorted years for a specific relation, with None at the end if present.

        Args:
            rel: The relation name.

        Returns:
            List of years (int or None) for this relation.
        """
        rel_years_raw = self.years_by_rel.get(rel, set())
        rel_years: list[int | None] = sorted(year for year in rel_years_raw if year is not None)
        if None in rel_years_raw:
            rel_years.append(None)
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
        period_accumulator: dict[tuple[RelationKey, int | None], dict]
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

    def _finalize_overall(self, overall_accumulator: dict[RelationKey, dict]) -> dict[RelationKey, dict]:
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
        period_align: str = "newest",
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


def _fetch_split_relation_rows(
    cursor,
    corpus: str,
    word: str,
    lemgram: bool,
    min_freq: int | None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> dict:
    """Fetch split (yearly) relation rows for a specific corpus.

    Args:
        cursor: Database cursor.
        corpus: The corpus name.
        word: The target word.
        lemgram: If `True`, select lemgrams; if `False`, select wordforms.
        min_freq: Minimum frequency filter or `None`.
        start_year: Start year (inclusive) or `None`.
        end_year: End year (inclusive) or `None`.

    Returns:
        A dictionary containing lists of relation rows.
    """
    triple_sql, triple_params = _build_split_triples_query(
        corpus,
        lemgram=lemgram,
        min_freq=min_freq,
        start_year=start_year,
        end_year=end_year,
    )
    cursor.execute(triple_sql, [word, *triple_params])
    triples = cursor.fetchall()

    head_sql, head_params = _build_head_query(corpus, lemgram=lemgram, start_year=start_year, end_year=end_year)
    cursor.execute(head_sql, [word, *head_params])
    heads = cursor.fetchall()

    dep_sql, dep_params = _build_dep_query(corpus, lemgram=lemgram, start_year=start_year, end_year=end_year)
    cursor.execute(dep_sql, [word, *dep_params])
    deps = cursor.fetchall()

    rel_sql, rel_params = _build_rel_query(corpus, start_year=start_year, end_year=end_year)
    cursor.execute(rel_sql, [*rel_params])
    rels = cursor.fetchall()

    return {
        "triples": triples,
        "heads": heads,
        "deps": deps,
        "rels": rels,
    }


def _fetch_overall_relation_rows(
    cursor,
    corpus: str,
    word: str,
    lemgram: bool,
    min_freq: int | None,
) -> dict[str, object]:
    """Fetch combined overall relation rows using query optimized for overall data.

    Args:
        cursor: Database cursor.
        corpus: The corpus name.
        word: The target word.
        lemgram: If `True`, select lemgrams; if `False`, select wordforms.
        min_freq: Minimum frequency filter or `None`.

    Returns:
        A dictionary containing relation rows and frequency maps.
    """
    sql, params = _build_overall_triples_query(corpus, lemgram=lemgram, min_freq=min_freq)
    cursor.execute(sql, [word, *params, word, *params])
    rows = cursor.fetchall()

    triples: list[dict[str, object]] = []
    # Maps for deduplicating head/dep/rel frequencies across rows
    head_rel_map: Counter[tuple[int, str, str]] = Counter()
    dep_rel_map: Counter[tuple[int, str, str, str]] = Counter()
    rel_map: Counter[str] = Counter()

    for row in rows:
        head_id = row["head"]
        dep_id = row["dep"]
        rel = row["rel"]
        head_pos = row["head_pos"]
        dep_pos = row["dep_pos"]
        dep_extra = row["dep_extra"]
        triples.append(
            {
                "role": row["role"],
                "relation_id": row["relation_id"],
                "rel": rel,
                "head": head_id,
                "head_string": row["head_string"],
                "head_extra": row["head_extra"],
                "head_pos": head_pos,
                "dep": dep_id,
                "dep_string": row["dep_string"],
                "dep_extra": dep_extra,
                "dep_pos": dep_pos,
                "freq": row["freq"],
                "corpus": row["corpus"],
            }
        )
        head_key = (head_id, head_pos, rel)
        dep_key = (dep_id, dep_pos, dep_extra, rel)
        # Head/dep/rel frequencies are deduplicated using maps
        head_rel_map[head_key] = row["head_rel_freq"]
        dep_rel_map[dep_key] = row["dep_rel_freq"]
        rel_map[rel] = row["rel_freq"]

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
            rel_name = str(row["rel"])
            counter_key = (rel_name, row["role"])
            counters[counter_key] += 1
            if counters[counter_key] > max_results:
                continue
            bucket_limited.append(row)
        limited.extend(bucket_limited)

    return limited


@bp.route("/relations", methods=["GET", "POST"])
@utils.main_handler
@utils.prevent_timeout
def relations(args: dict[str, Any], abort_event: gevent.event.Event | None = None) -> Iterator[dict]:
    """Calculate word picture data.

    Args:
        args: Dictionary of request arguments.
        abort_event: Optional event to signal abortion of processing.

    Yields:
        Dictionary containing the results or progress updates.
    """
    utils.assert_key("corpus", args, utils.IS_IDENT, True)
    utils.assert_key("word", args, "", True)
    utils.assert_key("type", args, r"(word|lemgram)", False)
    utils.assert_key("min", args, utils.IS_NUMBER, False)
    utils.assert_key("max", args, utils.IS_NUMBER, False)
    utils.assert_key("sort", args, r"(freq|freq_relative|mi|rmi)", False)
    utils.assert_key("incremental", args, r"(true|false)")
    utils.assert_key("split", args, r"(true|false)", False)
    utils.assert_key("period_size", args, utils.IS_NUMBER, False)
    utils.assert_key("period_align", args, r"(oldest|newest)", False)
    utils.assert_key("start_year", args, utils.IS_NUMBER, False)
    utils.assert_key("end_year", args, utils.IS_NUMBER, False)
    utils.assert_key("overall", args, r"(true|false)", False)
    utils.assert_key("max_scope", args, r"(overall|per_period)", False)

    corpora = utils.parse_corpora(args)
    utils.check_authorization(corpora)

    incremental = utils.parse_bool(args, "incremental", False)

    word = args["word"]
    lemgram: bool = args.get("type", "") == "lemgram"
    min_freq_value = args.get("min")
    min_freq = int(min_freq_value) if min_freq_value else None
    sort_field = args.get("sort") or "mi"
    max_results = int(args.get("max") or 15)
    period_size = max(int(args.get("period_size") or 1), 1)
    period_align = (args.get("period_align") or "newest").lower()
    start_year_value = args.get("start_year")
    start_year = int(start_year_value) if start_year_value not in {None, ""} else None
    end_year_value = args.get("end_year")
    end_year = int(end_year_value) if end_year_value not in {None, ""} else None
    include_split = utils.parse_bool(args, "split", False)
    include_overall = utils.parse_bool(args, "overall", True)
    max_scope = (args.get("max_scope") or "per_period").lower()
    # If false, time-sliced results are scoped to the overall selection instead of per-period limits
    limit_per_period = max_scope == "per_period"

    time_filter = start_year is not None or end_year is not None
    # Use split tables whenever split output or year filtering is requested
    use_split_data = include_split or time_filter

    if not include_split and not include_overall:
        yield {"ERROR": "Both split and overall results are disabled."}
        return

    overall_only = not include_split

    result = {}

    cursor = mysql.connection.cursor()
    cursor.execute("SET @@session.long_query_time = 1000;")

    # Get available tables
    cursor.execute("SHOW TABLES LIKE '" + app.config["DBWPTABLE"] + "_%_head_rel';")
    tables = {next(iter(r.values())) for r in cursor}

    # Filter out corpora which don't exist in database
    corpora = [
        c
        for c in corpora
        if app.config["DBWPTABLE"] + "_" + c.upper() + (f"{SPLIT_SUFFIX}_head_rel" if use_split_data else "_head_rel")
        in tables
    ]
    if not corpora:
        yield {"ERROR": "No word picture data available for the selected corpora."}
        return

    corpora_rest = corpora[:]
    corpus_results: dict[str, dict] = {}
    cache_prefixes = {}
    memcached_keys: dict[str, str] = {}
    cache_checksum = None

    if args["cache"]:
        cache_checksum = utils.get_hash((word, lemgram, min_freq_value, use_split_data, start_year, end_year))
        with memcached.get_client() as mc:
            cache_prefixes: dict[str, str] = utils.cache_prefix(mc, corpora)  # type: ignore
            for corpus in corpora:
                key = f"{cache_prefixes[corpus]}:relations_{cache_checksum}"
                memcached_keys[key] = corpus
            cached_data = mc.get_many(memcached_keys.keys())

        expected_keys = (
            {"triples", "heads", "deps", "rels"}
            if use_split_data
            else {"triples", "head_rel_map", "dep_rel_map", "rel_map"}
        )
        for key, data in (cached_data or {}).items():
            corpus_name = memcached_keys.get(key)
            if not corpus_name or not isinstance(data, dict):
                continue
            if not expected_keys <= data.keys():
                continue
            corpus_results[corpus_name] = data
            if corpus_name in corpora_rest:
                corpora_rest.remove(corpus_name)

    progress_index = 0

    # Fetch per-corpus rows from the chosen table family
    if corpora_rest:
        if incremental:
            yield {"progress_corpora": list(corpora_rest)}
        for corpus in corpora_rest:
            if abort_event and abort_event.is_set():
                cursor.close()
                return
            if not use_split_data:
                # Neither split output nor year filtering requested: use overall-optimized query
                data = _fetch_overall_relation_rows(cursor, corpus, word, lemgram, min_freq)
            else:
                data = _fetch_split_relation_rows(
                    cursor,
                    corpus,
                    word,
                    lemgram,
                    min_freq,
                    start_year=start_year,
                    end_year=end_year,
                )
            corpus_results[corpus] = data
            if args["cache"]:
                cache_key = f"{cache_prefixes[corpus]}:relations_{cache_checksum}"
                with memcached.get_client() as mc:
                    try:
                        mc.add(cache_key, data)
                    except MemcacheError:
                        pass
            if incremental:
                yield {f"progress_{progress_index}": {"corpus": corpus}}
                progress_index += 1

    cursor.close()

    # Get yearly size of corpora to be able to compute relative frequencies
    corpus_timedata = utils.generator_to_dict(
        timespan.timespan(args={"corpus": corpora, "granularity": "y", "cache": args["cache"]}, no_combined_cache=True)
    )
    # Sum up total frequencies per year
    corpus_size_per_year: dict[int | None, int] = defaultdict(int)
    for corpus in corpus_timedata["corpora"]:
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

    # Fast path: overall-only with no year filtering
    if overall_only and not use_split_data:
        relations_rows = _build_overall_only_relations(
            corpora,
            corpus_results,
            sort_field=sort_field,
            max_results=max_results,
            corpus_size=total_corpus_size,
        )
        result["relations"] = relations_rows
        yield result
        return

    # Everything past this point uses the accumulator for split/overall data

    # Aggregate split rows for overall + time-sliced outputs
    acc = _WordPictureAccumulator()
    for corpus in corpora:
        if not (corpus_data := corpus_results.get(corpus)):
            continue
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

    overall_relation_entries: list[dict[str, object]] = []
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

    selected_entries: list[dict[str, object]] = []
    selected_keys: list[tuple] = []

    if overall_relation_entries:
        # We have overall entries, meaning either overall results were requested or they are needed for scoping
        # time-sliced results.

        # Sort overall entries by relation and the chosen sort field, then apply max_results per relation and role.
        overall_relation_entries.sort(
            key=lambda entry: (entry["rel"], entry.get(sort_field, entry["mi"])), reverse=True
        )
        counters: Counter[tuple[object, str]] = Counter()
        for entry in overall_relation_entries:
            key = (entry["rel"], entry["role"])
            counters[key] += 1
            if max_results and counters[key] > max_results:
                continue
            selected_entries.append(entry)
        if include_overall:
            result["relations"] = [_relation_output(entry) for entry in selected_entries]
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
                per_period_rows = _limit_rows_per_bucket(per_period_rows, "period_start", sort_field, max_results)
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
                key = (
                    f"{row['period_start']}-{row['period_end']}"
                    if period_size > 1
                    else str(row["period_start"] if row["period_start"] is not None else "")
                )
                grouped_time_result.setdefault(key, []).append(_relation_output(row))
            result["relations_time"] = grouped_time_result
        else:
            result["relations_time"] = {}

    yield result


@bp.route("/relations_time", methods=["GET", "POST"])
@utils.main_handler
@utils.prevent_timeout
def relations_time(args: dict[str, Any]) -> Iterator[dict]:
    """Calculate word picture data with time splits.

    Args:
        args: Dictionary of request arguments.

    Yields:
        Dictionary containing the results.
    """
    # Reuse the relations function with specific parameters
    args["overall"] = args.get("overall", "false")
    args["split"] = "true"
    yield from relations(args)


@bp.route("/relations_sentences", methods=["GET", "POST"])
@utils.main_handler
@utils.prevent_timeout
def relations_sentences(args: dict[str, Any]) -> Iterator[dict]:
    """Execute a CQP query to find sentences with a given relation from a word picture.

    Args:
        args: Dictionary of request arguments.

    Yields:
        Dictionary containing the results.
    """
    utils.assert_key("source", args, "", True)
    utils.assert_key("start", args, utils.IS_NUMBER, False)
    utils.assert_key("end", args, utils.IS_NUMBER, False)
    utils.assert_key("split", args, r"(true|false)", False)

    temp_source = args.get("source", [])
    if isinstance(temp_source, str):
        temp_source = temp_source.split(utils.QUERY_DELIM)
    source = defaultdict(set)
    for s in temp_source:
        c, i = s.split(":")
        source[c].add(i)

    utils.check_authorization(source.keys())

    yearly = utils.parse_bool(args, "split", False)
    table_suffix = f"{SPLIT_SUFFIX}_sentences" if yearly else "_sentences"

    start = int(args.get("start") or 0)
    end = int(args.get("end") or 9)
    shown = args.get("show") or "word"
    shown_structs = args.get("show_struct") or []
    if isinstance(shown_structs, str):
        shown_structs = shown_structs.split(utils.QUERY_DELIM)
    shown_structs = set(shown_structs)

    default_context = args.get("default_context") or "1 sentence"

    querystarttime = time.time()

    cursor = mysql.connection.cursor()
    cursor.execute("SET @@session.long_query_time = 1000;")
    selects = []
    counts = []

    # Get available tables
    cursor.execute(f"SHOW TABLES LIKE '{app.config['DBWPTABLE']}_%{table_suffix}';")
    tables = {next(iter(r.values())) for r in cursor}
    # Filter out corpora which don't exist in database
    source = sorted(
        [c for c in iter(source.items()) if f"{app.config['DBWPTABLE']}_{c[0].upper()}{table_suffix}" in tables]
    )
    if not source:
        yield {}
        return
    corpora = [x[0] for x in source]

    for s in source:
        corpus, ids = s
        ids = [int(i) for i in ids]
        ids_list = "(" + ", ".join(f"{i:d}" for i in ids) + ")"

        corpus_table_sentences = app.config["DBWPTABLE"] + f"_{corpus.upper()}{table_suffix}"

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

    sql_count = " UNION ALL ".join(counts)
    cursor.execute(sql_count)

    corpus_hits = {}
    for row in cursor:
        corpus_hits[row["corpus"]] = int(row["freq"])

    sql = " UNION ALL ".join(selects) + (f" LIMIT {start}, {end - start + 1}")
    cursor.execute(sql)

    querytime = time.time() - querystarttime
    corpora_dict = {}
    for row in cursor:
        corpora_dict.setdefault(row["corpus"], {}).setdefault(row["sentence"], []).append((row["start"], row["end"]))

    cursor.close()

    total_hits = sum(corpus_hits.values())

    if not corpora_dict:
        yield {"hits": 0}
        return

    cqpstarttime = time.time()
    result = {}

    for corp, sids in sorted(corpora_dict.items(), key=operator.itemgetter(0)):
        cqp = '<sentence_id="{}"> []* </sentence_id> within sentence'.format("|".join(set(sids.keys())))
        q = {
            "cqp": cqp,
            "corpus": corp,
            "start": "0",
            "end": str(end - start),
            "show_struct": ["sentence_id", *shown_structs],
            "default_context": default_context,
        }
        if shown:
            q["show"] = shown
        result_temp = utils.generator_to_dict(query.query(q))

        # Loop backwards since we might be adding new items
        for i in range(len(result_temp["kwic"]) - 1, -1, -1):
            s = result_temp["kwic"][i]
            sid = s["structs"]["sentence_id"]
            r = sids[sid][0]
            sentence_start = s["match"]["start"]
            s["match"]["start"] = sentence_start + min(map(int, r)) - 1
            s["match"]["end"] = sentence_start + max(map(int, r))

            # If the same relation appears more than once in the same sentence,
            # append copies of the sentence as separate results
            for r in sids[sid][1:]:
                s2 = deepcopy(s)
                s2["match"]["start"] = sentence_start + min(map(int, r)) - 1
                s2["match"]["end"] = sentence_start + max(map(int, r))
                result_temp["kwic"].insert(i + 1, s2)

        result.setdefault("kwic", []).extend(result_temp["kwic"])

    result["hits"] = total_hits
    result["corpus_hits"] = corpus_hits
    result["corpus_order"] = corpora
    result["querytime"] = querytime
    result["cqptime"] = time.time() - cqpstarttime

    yield result


@bp.route("/relations_time_sentences", methods=["GET", "POST"])
@utils.main_handler
@utils.prevent_timeout
def relations_time_sentences(args: dict[str, Any]) -> Iterator[dict]:
    """Execute a CQP query to find sentences with a given relation from a word picture.

    Args:
        args: Dictionary of request arguments.

    Yields:
        Dictionary containing the results.
    """
    # Reuse the relations_sentences function with specific parameters
    args["split"] = "true"
    yield from relations_sentences(args)

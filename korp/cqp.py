"""CQP query processing, parsing, and optimization."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Sequence
from enum import Enum

# Special symbols used when parsing CQP results; should not appear in corpus data
END_OF_LINE = "-::-EOL-::-"
LEFT_DELIM = "---:::"
RIGHT_DELIM = ":::---"

UNDEF_VALUE = "__UNDEF__"


class CQPError(Exception):
    """Custom exception for CQP errors."""


def translate_undef(s: str | None) -> str | None:
    """Translate '__UNDEF__' to None.

    '__UNDEF__' can be used in corpora to represent undefined values.

    Args:
        s: The input string.

    Returns:
        `None` if the input string is '__UNDEF__', otherwise the original string.
    """
    return None if s == UNDEF_VALUE else s


def parse_within(within: Sequence[str] | None, default_within: str | None = None) -> dict[str, str | None]:
    """Parse 'within' parameter into a dictionary mapping corpora to within values.

    Args:
        within: A sequence of 'CORPUS:WITHIN' pairs.
        default_within: The default within value to use for corpora not specified in the 'within' parameter.

    Returns:
        A dictionary mapping corpus names to their respective within values.

    Raises:
        ValueError: If the 'within' parameter is malformed.
    """
    within_dict = defaultdict(lambda: default_within)
    within = within or []

    for pair in within:
        if ":" not in pair:
            raise ValueError("Malformed value for key 'within'.")
        corpus, within_value = pair.split(":", 1)
        within_dict[corpus.upper()] = within_value
    return within_dict


def parse_cqp(cqp: str) -> tuple[list[str], bool]:
    """Try to parse a CQP query, returning identified tokens and a boolean indicating partial failure if True.

    This is used by the query optimizer, and by "free order" searches.

    Args:
        cqp: The CQP query string.

    Returns:
        A tuple containing:
            - A list of strings representing the identified tokens.
            - A boolean indicating whether the parsing was only partially successful.
    """
    cqp_len = len(cqp)
    sections: list[list[int]] = []
    last_start = 0
    in_bracket = False
    in_quote = False
    in_curly = False
    escaping = False
    quote_type = ""

    for i, c in enumerate(cqp):
        # Handle escape sequences (only relevant inside quotes)
        if escaping:
            escaping = False
            continue

        if in_quote:
            if c == "\\":
                escaping = True
            elif c == quote_type:
                if i + 1 < cqp_len and cqp[i + 1] == quote_type:
                    # Quote escaped by doubling
                    escaping = True
                else:
                    # End of a quote
                    in_quote = False
                    if not in_bracket:
                        sections.append([last_start, i])
            # Skip all bracket/curly checks when inside a quote
            continue

        # Outside quotes
        if c in "'\"":
            # Beginning of a quote
            in_quote = True
            quote_type = c
            if not in_bracket:
                last_start = i
        elif c == "[":
            if not in_bracket:
                # Beginning of a token
                last_start = i
                in_bracket = True
                if i + 1 < cqp_len and cqp[i + 1] == ":":
                    # Zero-width assertion encountered, which cannot be handled by MU query
                    return [], True
        elif c == "]":
            if in_bracket:
                # End of a token
                sections.append([last_start, i])
                in_bracket = False
        elif c == "{" and not in_bracket:
            in_curly = True
        elif c == "}" and not in_bracket and in_curly:
            in_curly = False
            sections[-1][1] = i

    # Build token list and detect non-token content ("rest") between sections
    sections.append([cqp_len, cqp_len])
    tokens: list[str] = []
    rest = False
    prev_end = 0

    for start, end in sections:
        if prev_end < start and cqp[prev_end + 1 : start].strip():
            rest = True
        prev_end = end
        token = cqp[start : end + 1]
        if token:
            tokens.append(token)

    return tokens, rest


def make_cqp(cqp: str, within: str | None = None, cut: str | None = None, expand: str | None = None) -> str:
    """Combine CQP query and extra options into a single CQP query string.

    Args:
        cqp: The CQP query string.
        within: The 'within' option.
        cut: The 'cut' option.
        expand: The 'expand' option.

    Returns:
        The combined CQP query string with options appended, and terminated with a semicolon.
    """
    parts = [cqp]
    if within:
        parts.append(f"within {within}")
    if cut:
        parts.append(f"cut {cut}")
    if expand:
        parts.append(f"expand {expand}")
    return " ".join(parts) + ";"


def make_query(cqp: str | list[str]) -> list[str]:
    """Create web-safe commands for a CQP query.

    This wraps the CQP query with commands to enable and disable query lock mode. This prevents execution of arbitrary
    commands, allowing only queries to be executed.

    Args:
        cqp: The CQP query string or list of CQP query strings. Each string must be terminated with a semicolon.

    Returns:
        A list of CQP commands with query lock enabled.
    """
    querylock = random.randrange(10**8, 10**9)
    if isinstance(cqp, str):
        cqp = [cqp]
    return [f"set QueryLock {querylock};", *cqp, f"unlock {querylock};"]


# Pre-compiled patterns for wildcard/repetition parsing in query_optimize
_RE_WILDCARD_RANGE = re.compile(r"\{\s*(\d+)\s*,\s*(\d*)\s*\}$")
_RE_WILDCARD_EXACT = re.compile(r"\{\s*(\d*)\s*\}$")
_RE_REPETITION = re.compile(r"\{.*?\}$")
_WILDCARD_MAX = 9999  # Upper bound representing an unbounded wildcard range


def _parse_wildcard_repeat(token: str) -> tuple[int, int] | None:
    """Parse repetition counts from a wildcard token like `[]{2,5}`.

    Args:
        token: A wildcard token string (must start with `[]`).

    Returns:
        A (min, max) tuple, or None if the token has no parseable repetition.
    """
    if token == "[]":
        return 1, 1
    if m := _RE_WILDCARD_RANGE.search(token):
        return int(m.group(1)), int(m.group(2)) if m.group(2) else _WILDCARD_MAX
    if m := _RE_WILDCARD_EXACT.search(token):
        n = int(m.group(1))
        return n, n
    return None


class QueryOptimizeResult(Enum):
    """Result codes for query optimization."""

    SUCCESS = 0
    """Optimization successful; the query was transformed into an optimized MU query."""

    NOT_NEEDED = 1
    """Optimization not needed; the query is too simple to benefit from optimization (e.g., single word search)."""

    NOT_POSSIBLE = 2
    """Optimization not possible; the query contains constructs that prevent optimization (e.g., repetition of
    non-wildcards)."""


def optimize_query(
    cqp: str, cqp_params: dict, find_match: bool = True, expand: bool = True, free_search: bool = False
) -> tuple[QueryOptimizeResult, list[str]]:
    """Optimize simple queries with multiple words by converting them to MU queries.

    Optimization only works for queries with at least two tokens, or one token preceded by one or more wildcards. The
    query also must use `within`.

    Args:
        cqp: The CQP query string.
        cqp_params: Additional CQP parameters (within, cut, expand).
        find_match: Whether to mark all matching words in the result (not just the first).
        expand: Whether to expand the query.
        free_search: Whether the query is a free order search.

    Returns:
        A tuple containing:
        - A QueryOptimizeResult indicating the optimization outcome.
        - A list of strings representing the optimized query.

    Raises:
        CQPError: If the query cannot be optimized due to unsupported constructs.
    """
    tokens, rest = parse_cqp(cqp)
    within = cqp_params.get("within")
    fallback_query = make_query(make_cqp(cqp, **cqp_params))

    leading_wildcards = False

    if free_search:
        # Don't allow wildcards in free order queries
        if any(token.startswith("[]") for token in tokens):
            raise CQPError("Wildcards not allowed in free order queries.")

        # Don't allow distance-based within values in free order queries (e.g. "5 sentence")
        if within and re.match(r"^\d+ ", within):
            raise CQPError("Distance-based 'within' values not allowed in free order queries.")
    else:
        # Strip leading and trailing wildcards since they only slow things down
        start = 0
        while start < len(tokens) and tokens[start].startswith("[]"):
            leading_wildcards = True
            start += 1
        end = len(tokens)
        while end > start and tokens[end - 1].startswith("[]"):
            end -= 1
        tokens = tokens[start:end]

    if not tokens or (len(tokens) == 1 and not leading_wildcards):
        # Query doesn't benefit from optimization
        return QueryOptimizeResult.NOT_NEEDED, fallback_query
    if rest or not within:
        # Couldn't optimize this query
        return QueryOptimizeResult.NOT_POSSIBLE, fallback_query

    # Build the MU command
    mu_parts: list[str] = ["MU"]
    wildcards: dict[int, tuple[int, int]] = {}

    for i, token in enumerate(tokens[:-1]):
        if token.startswith("[]"):
            repeat = _parse_wildcard_repeat(token)
            if repeat is not None:
                wildcards[i] = repeat
            continue
        if _RE_REPETITION.search(token):
            # Repetition for anything other than wildcards can't be optimized
            return QueryOptimizeResult.NOT_POSSIBLE, fallback_query
        mu_parts.append(f"(meet {token}")

    if _RE_REPETITION.search(tokens[-1]):
        return QueryOptimizeResult.NOT_POSSIBLE, fallback_query

    mu_parts.append(tokens[-1])

    # Build closing parts with distance constraints (reverse order)
    wc_min = wc_max = 1
    for i in range(len(tokens) - 2, -1, -1):
        if i in wildcards:
            wc_min += wildcards[i][0]
            wc_max += wildcards[i][1]
            continue
        if i + 1 in wildcards:
            mu_parts.append(f"{within})" if wc_max >= _WILDCARD_MAX else f"{wc_min} {wc_max})")
            wc_min = wc_max = 1
        elif free_search:
            mu_parts.append(f"{within})")
        else:
            mu_parts.append("1 1)")

    mu_cmd = " ".join(mu_parts)
    cmd: list[str] = []

    if find_match and not free_search:
        # MU searches only highlight the first keyword of each hit. To highlight all keywords we need to
        # do a new non-optimized search within the results, and to be able to do that we first need to expand the rows.
        # Most of the time we only need to expand to the right, except for when leading wildcards are used.
        direction = "expand to" if leading_wildcards else "expand right to"
        cmd.extend([f"{mu_cmd} {direction} {within};", "Last;", *fallback_query])
    elif expand or free_search:
        cmd.append(f"{mu_cmd} expand to {within};")
    else:
        cmd.append(f"{mu_cmd};")

    return QueryOptimizeResult.SUCCESS, cmd

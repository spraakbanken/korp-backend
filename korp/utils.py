"""General utility functions and classes."""

from __future__ import annotations

import datetime
import hashlib
from collections.abc import AsyncGenerator, Generator, Iterable
from types import SimpleNamespace
from typing import Any

from korp.db import escape_string as _db_escape_string

QUERY_DELIM = ","


def get_hash(values: Iterable[Any]) -> str:
    """Get a hash for a list of values.

    Args:
        values: A list of values to hash.

    Returns:
        A SHA-256 hash of the concatenated values.
    """
    return hashlib.sha256(";".join(v if isinstance(v, str) else str(v) for v in values).encode()).hexdigest()


class Namespace(SimpleNamespace):
    """Simple namespace class to hold attributes."""


def split_csv(values: str | Iterable[str] | None) -> list[str]:
    """Split comma-separated values into a list.

    Accepts a string (comma-separated) or an iterable of strings (repeated query params).
    Empty values are dropped. Order is preserved.

    Args:
        values: A string of comma-separated values, an iterable of strings, or None.

    Returns:
        A list of individual values.
    """
    if values is None:
        return []

    raw_values = [values] if isinstance(values, str) else list(values)
    return [item for raw in raw_values for item in raw.split(QUERY_DELIM) if item]


def strptime(date: str) -> datetime.datetime:
    """Take a date in string format and return a datetime object.

    We need this since the built-in strptime isn't thread safe (and this is much faster).

    Args:
        date: Date string in the format "YYYYMMDDhhmmss".

    Returns:
        A datetime object representing the parsed date.
    """
    year = int(date[:4])
    month = int(date[4:6]) if len(date) > 4 else 1  # noqa: PLR2004
    day = int(date[6:8]) if len(date) > 6 else 1  # noqa: PLR2004
    hour = int(date[8:10]) if len(date) > 8 else 0  # noqa: PLR2004
    minute = int(date[10:12]) if len(date) > 10 else 0  # noqa: PLR2004
    second = int(date[12:14]) if len(date) > 12 else 0  # noqa: PLR2004
    return datetime.datetime(year, month, day, hour, minute, second)


def sql_escape(s: str) -> str:
    """Return SQL-escaped version of string s."""
    return _db_escape_string(s) if isinstance(s, str) else s


def sync_generator_to_dict(generator: Generator[dict, None, None]) -> dict:
    """Convert a sync generator yielding dicts to a single dict.

    Args:
        generator: Generator yielding dicts.

    Returns:
        A single dict containing all key-value pairs from the yielded dicts.
    """
    result: dict = {}
    for d in generator:
        result.update(d)
    return result


async def async_generator_to_dict(generator: AsyncGenerator[dict, None]) -> dict:
    """Convert an async generator yielding dicts to a single dict.

    Args:
        generator: Generator yielding dicts.

    Returns:
        A single dict containing all key-value pairs from the yielded dicts.
    """
    result = {}
    async for d in generator:
        if isinstance(d, dict):
            result.update(d)
    return result

"""Query translation plugin interface and request resolver."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, Protocol, Self, TypeVar

from pydantic import model_validator

from korp.api import params
from korp.api.requests import RequestModel
from korp.dependencies import Ctx
from korp.handler import APIValidationError


class QueryTranslator(Protocol):
    """A translation provider instantiated by a plugin's `QUERY_TRANSLATOR_FACTORY`."""

    languages: frozenset[str]
    """A set of supported languages for the query translator."""

    async def translate(self, ctx: Ctx, language: str, query: str) -> str:
        """Translate one query into a CQP expression."""
        ...


def register_provider(
    registry: dict[str, QueryTranslator],
    factory: Callable[[Mapping[str, Any]], QueryTranslator],
    config: Mapping[str, Any],
) -> None:
    """Instantiate and register a provider, rejecting invalid or duplicate exports.

    Raises:
        TypeError: If the factory or provider does not implement the contract.
        ValueError: If a language is invalid or already registered.
    """
    if not callable(factory):
        raise TypeError("QUERY_TRANSLATOR_FACTORY must be callable")
    provider = factory(deepcopy(dict(config)))
    languages = getattr(provider, "languages", None)
    if not isinstance(languages, frozenset) or not languages or not callable(getattr(provider, "translate", None)):
        raise TypeError(
            "Query translators must expose a non-empty frozenset of languages and an async translate method"
        )
    if not inspect.iscoroutinefunction(provider.translate):
        raise TypeError("Query translator translate must be async")
    for language in languages:
        if not isinstance(language, str) or not language or language.strip() != language or language == "cqp":
            raise ValueError(f"Invalid query translator language: {language!r}")
        if language in registry:
            raise ValueError(f"Duplicate query translator language: {language!r}")
    registry.update(dict.fromkeys(languages, provider))


def validate_query_input(raw_cqp: object, query: object, language: object, *, prefix: str = "") -> None:
    """Require exactly one of raw CQP or query+language.

    Raises:
        ValueError: If the input selection is invalid.
    """
    if raw_cqp is not None:
        if query is not None or language is not None:
            raise ValueError(f"Use either '{prefix}cqp' or '{prefix}query' with '{prefix}query_language'")
    elif query is None or language is None:
        raise ValueError(f"Provide '{prefix}cqp' or both '{prefix}query' and '{prefix}query_language'")


class SearchQueryRequest(RequestModel):
    """Shared mutually exclusive inputs for searches supporting translation."""

    corpora: params.CorporaParam
    cqp_query: params.OptionalCQPParam = None
    query: params.TranslationQueriesParam = None
    query_language: params.QueryLanguagesParam = None

    @model_validator(mode="after")
    def validate_query(self) -> Self:
        """Validate input selection.

        Returns:
            The validated request.

        Raises:
            ValueError: If the language list does not match the query stages.
        """
        validate_query_input(self.cqp_query, self.query, self.query_language)
        if (
            self.query is not None
            and self.query_language is not None
            and len(self.query_language) not in {1, len(self.query)}
        ):
            raise ValueError("Supply one query_language for all queries or one language per query")
        return self


async def translate_query(ctx: Ctx, language: str, query: str, *, prefix: str = "") -> str:
    """Translate one query into a CQP expression using the app's registered provider.

    Returns:
        The translated CQP expression.

    Raises:
        APIValidationError: If translation is unavailable or the provider rejects input.
        TypeError: If a provider returns an invalid result.
    """
    provider = ctx.request.app.state.query_translators.get(language)
    if provider is None:
        raise APIValidationError(f"Query language is not enabled: {language}", field=f"{prefix}query_language")
    try:
        result = await provider.translate(ctx, language, query)
    except ValueError as exc:
        raise APIValidationError(str(exc), field=f"{prefix}query") from exc
    if not isinstance(result, str) or not result.strip():
        raise TypeError("Query translators must return a non-empty CQP expression")
    return result


SearchRequestT = TypeVar("SearchRequestT", bound=SearchQueryRequest)


async def resolve_search_request(ctx: Ctx, request: SearchRequestT) -> SearchRequestT:
    """Resolve a request with alternative queries into one with CQP.

    Returns:
        The original CQP request or a resolved copy of a translated request.

    Raises:
        APIValidationError: If a stage fails translation or free-order input is supplied.
    """
    if request.query is None:
        return request
    assert request.query_language is not None
    if getattr(request, "free_order", False):
        raise APIValidationError("Alternative queries do not support free_order=true.", field="free_order")
    translated = []
    for index, query in enumerate(request.query):
        language_index = 0 if len(request.query_language) == 1 else index
        language = request.query_language[language_index]
        try:
            translated.append(await translate_query(ctx, language, query))
        except APIValidationError as exc:
            field = exc.field
            if field == "query":
                field = f"query[{index}]"
            elif field == "query_language":
                field = f"query_language[{language_index}]"
            raise APIValidationError(f"Query stage {index + 1}: {exc.detail}", field=field) from exc
    return request.model_copy(update={"cqp_query": translated, "query": None, "query_language": None})

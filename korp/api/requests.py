"""Shared request models for JSON API operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

RequestModelT = TypeVar("RequestModelT", bound="RequestModel")


class RequestModel(BaseModel):
    """Base model for strict JSON request bodies."""

    # Enable `validate_by_name` to allow `QueryRequestModel.to_request()` to pass field names rather than aliases.
    model_config = ConfigDict(extra="forbid", validate_by_alias=True, validate_by_name=True)

    # Fields that must be JSON arrays
    json_array_fields: ClassVar[frozenset[str]] = frozenset()

    # Fields that must be JSON arrays, but may contain commas
    json_array_fields_allowing_commas: ClassVar[frozenset[str]] = frozenset()

    # Whether to require JSON arrays for fields documented as such. This is disabled for GET query models.
    require_json_array_input: ClassVar[bool] = True

    @model_validator(mode="before")
    @classmethod
    def require_json_arrays(cls, value: Any) -> Any:
        """Require that declared JSON array fields are actually arrays.

        Unless also declared in `json_array_fields_allowing_commas`, this validator will reject any array field that
        contains a string with a comma, since that is likely to be a CSV value that should have been split into multiple
        array items.

        Returns:
            The unchanged request mapping after array validation.

        Raises:
            ValueError: If an array field contains a scalar value.
        """
        if not cls.require_json_array_input or not isinstance(value, Mapping):
            return value

        for field_name in cls.json_array_fields:
            field = cls.model_fields[field_name]
            input_name = field.alias or field_name
            field_value = value.get(input_name, value.get(field_name))
            if field_value is not None and not isinstance(field_value, (list, tuple)):
                raise ValueError(f"'{input_name}' must be a JSON array")
            if field_value is not None and any(not isinstance(item, str) for item in field_value):
                raise ValueError(f"'{input_name}' must contain strings")
            if (
                field_name not in cls.json_array_fields_allowing_commas
                and field_value is not None
                and any(isinstance(item, str) and "," in item for item in field_value)
            ):
                raise ValueError(f"'{input_name}' must contain one value per JSON array item")
        return value


class CommonQueryControls(BaseModel):
    """Response controls shared by API query strings."""

    model_config = ConfigDict(extra="forbid")

    cache: bool = Field(True, description="Whether to use caching for the request.")
    debug: bool = Field(False, description="Whether to include debug information in responses.")
    indent: Annotated[int, Field(ge=0, le=16)] = Field(0, description="Number of spaces to indent JSON output.")
    stream: bool = Field(
        False,
        description=(
            "Whether to return an `application/x-ndjson` event stream instead of one JSON object. Each line is a "
            "complete `progress`, `result`, `error`, `keepalive`, or `complete` event. Not every route emits "
            "progress events."
        ),
    )


class QueryRequestModel(CommonQueryControls, RequestModel):
    """Base for GET query models with support for route-specific CSV fields."""

    # Disable JSON array validation.
    require_json_array_input = False

    # Fields that support CSV input. These fields will be split on commas before validation.
    csv_fields: ClassVar[frozenset[str]] = frozenset()

    @model_validator(mode="before")
    @classmethod
    def normalize_csv_fields(cls, value: Any) -> Any:
        """Split declared CSV fields into arrays before validation.

        Returns:
            A copy of the input mapping with declared CSV fields normalized.
        """
        if not isinstance(value, Mapping):
            return value

        normalized = dict(value)
        for field_name in cls.csv_fields:
            field = cls.model_fields[field_name]
            input_name = field.alias or field_name
            if input_name not in normalized and field_name not in normalized:
                continue
            key = input_name if input_name in normalized else field_name
            raw_value = normalized[key]
            values = raw_value if isinstance(raw_value, (list, tuple)) else [raw_value]
            normalized[key] = [part for item in values for part in str(item).split(",") if part]
        return normalized

    def to_request(self, model: type[RequestModelT]) -> RequestModelT:
        """Convert this parsed GET query into the corresponding request model.

        GET query models also contain the query-string-only controls from `CommonQueryControls`. This method keeps only
        the fields that are part of the request model, and validates and returns the same data as the corresponding
        request model would have.

        Args:
            model: The request model to create.

        Returns:
            A validated request model containing only the fields that are part of the request model.
        """
        values = {field_name: getattr(self, field_name) for field_name in model.model_fields}
        return model.model_validate(values)

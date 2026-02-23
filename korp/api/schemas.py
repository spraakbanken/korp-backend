"""Pydantic response/request models."""

from typing import Any

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema


class CommonResponse(BaseModel):
    """Common response model."""

    DEBUG: dict[str, Any] | SkipJsonSchema[None] = Field(
        None, description="Debug information, included only if debug mode is enabled."
    )
    time: float = Field(..., description="Time taken to process the request in seconds.", examples=[0.123])

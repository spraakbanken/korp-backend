"""Pydantic response/request models."""

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema


class ResponseModel(BaseModel):
    """Base class for documented API response structures."""

    model_config = ConfigDict(extra="forbid")


class CommonResponse(ResponseModel):
    """Common response model."""

    debug: dict[str, Any] | SkipJsonSchema[None] = Field(
        None, description="Debug information, included only if debug mode is enabled."
    )
    elapsed: float = Field(..., description="Time taken to process the request in seconds.", examples=[0.123])


class ErrorResponse(ResponseModel):
    """Problem Details-style error returned by the API."""

    code: str = Field(..., description="Stable Korp machine-readable error identifier.")
    title: str = Field(..., description="Short human-readable summary of the problem type.")
    status: int = Field(..., ge=400, le=599, description="HTTP status associated with the error.")
    detail: str = Field(..., description="Human-readable details about this occurrence of the problem.")
    field: str | SkipJsonSchema[None] = Field(None, description="Related request field, when known.")
    errors: list[dict[str, Any]] | SkipJsonSchema[None] = Field(
        None, description="Structured parameter errors supplied when request validation fails."
    )
    traceback: list[str] | SkipJsonSchema[None] = Field(
        None,
        description=(
            "Traceback lines, included only when server-side traceback exposure and request debug mode are enabled."
        ),
    )


class StreamProgressEvent(ResponseModel):
    """Progress made while producing a streamed response."""

    event: Literal["progress"]
    completed: int = Field(..., ge=0, description="Number of completed work items.")
    total: int = Field(..., ge=0, description="Total number of expected work items.")
    corpus: str | None = Field(None, description="Corpus whose work item completed.")
    corpora: list[str] | None = Field(None, description="Corpora covered by the operation, present in initial events.")
    hits: int | None = Field(None, ge=0, description="Hits found for the completed corpus, when known.")


class StreamResultEvent(ResponseModel):
    """One result fragment from a streamed response."""

    event: Literal["result"]
    data: dict[str, Any] = Field(
        ...,
        min_length=1,
        description=(
            "Non-empty fragment of the ordinary JSON response. Each top-level key may occur in only one result event. "
            "The `elapsed` key is reserved for the complete event."
        ),
    )


class StreamErrorEvent(ResponseModel):
    """Failure encountered after a streamed response started."""

    event: Literal["error"]
    error: ErrorResponse


class StreamCompleteEvent(ResponseModel):
    """Final event in a streamed response."""

    event: Literal["complete"]
    ok: bool = Field(..., description="Whether the operation completed without a streamed error.")
    elapsed: float = Field(..., ge=0, description="Time taken to process the request in seconds.")


class StreamKeepaliveEvent(ResponseModel):
    """Keepalive emitted while a streamed response is otherwise idle."""

    event: Literal["keepalive"]


StreamEvent: TypeAlias = Annotated[
    StreamProgressEvent | StreamResultEvent | StreamErrorEvent | StreamCompleteEvent | StreamKeepaliveEvent,
    Field(discriminator="event"),
]

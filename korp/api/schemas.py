"""Pydantic response/request models."""

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema


class CommonResponse(BaseModel):
    """Common response model."""

    debug: dict[str, Any] | SkipJsonSchema[None] = Field(
        None, description="Debug information, included only if debug mode is enabled."
    )
    elapsed: float = Field(..., description="Time taken to process the request in seconds.", examples=[0.123])
    error: str | dict[str, Any] | SkipJsonSchema[None] = Field(
        None, description="Error message or details, included only if an error occurred."
    )


class StreamProgressEvent(BaseModel):
    """Progress made while producing a streamed response."""

    model_config = ConfigDict(extra="forbid")

    event: Literal["progress"]
    completed: int = Field(..., ge=0, description="Number of completed work items.")
    total: int = Field(..., ge=0, description="Total number of expected work items.")
    corpus: str | None = Field(None, description="Corpus whose work item completed.")
    corpora: list[str] | None = Field(None, description="Corpora covered by the operation, present in initial events.")
    hits: int | None = Field(None, ge=0, description="Hits found for the completed corpus, when known.")


class StreamResultEvent(BaseModel):
    """One result fragment from a streamed response."""

    model_config = ConfigDict(extra="forbid")

    event: Literal["result"]
    data: dict[str, Any] = Field(..., description="Result fragment to merge with preceding result fragments.")


class StreamErrorDetails(BaseModel):
    """Error details for a streamed response."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., description="Machine-readable error type.")
    value: str = Field(..., description="Human-readable error message.")
    traceback: list[str] | None = Field(None, description="Traceback lines, included only in debug mode.")


class StreamErrorEvent(BaseModel):
    """Failure encountered after a streamed response started."""

    model_config = ConfigDict(extra="forbid")

    event: Literal["error"]
    error: StreamErrorDetails


class StreamCompleteEvent(BaseModel):
    """Final event in a streamed response."""

    model_config = ConfigDict(extra="forbid")

    event: Literal["complete"]
    ok: bool = Field(..., description="Whether the operation completed without a streamed error.")
    elapsed: float = Field(..., ge=0, description="Time taken to process the request in seconds.")


class StreamKeepaliveEvent(BaseModel):
    """Keepalive emitted while a streamed response is otherwise idle."""

    model_config = ConfigDict(extra="forbid")

    event: Literal["keepalive"]


StreamEvent: TypeAlias = Annotated[
    StreamProgressEvent | StreamResultEvent | StreamErrorEvent | StreamCompleteEvent | StreamKeepaliveEvent,
    Field(discriminator="event"),
]

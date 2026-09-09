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


class ErrorDetails(ResponseModel):
    """Error details returned as part of a JSON response."""

    type: str = Field(..., description="Machine-readable error type.")
    value: str = Field(..., description="Human-readable error message.")
    traceback: list[str] | SkipJsonSchema[None] = Field(
        None, description="Traceback lines, included only in debug mode."
    )


class LateJsonErrorResponse(ResponseModel):
    """Error returned after an ordinary JSON response has already started."""

    error: ErrorDetails
    elapsed: float = Field(..., description="Time taken before the request failed, in seconds.", examples=[0.123])


class PreflightErrorResponse(ResponseModel):
    """Error returned before the response body starts."""

    error: ErrorDetails


class HTTPErrorResponse(ResponseModel):
    """Error returned by FastAPI for an HTTPException before streaming starts."""

    detail: Any = Field(..., description="HTTP exception details.")


class RequestValidationErrorResponse(ResponseModel):
    """Error raised while FastAPI validates a request or dependency."""

    detail: Any = Field(..., description="FastAPI request-validation or HTTP-exception details.")


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
    data: dict[str, Any] = Field(..., description="Result fragment to merge with preceding result fragments.")


class StreamErrorDetails(ErrorDetails):
    """Error details for a streamed response."""


class StreamErrorEvent(ResponseModel):
    """Failure encountered after a streamed response started."""

    event: Literal["error"]
    error: StreamErrorDetails


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

"""Canonical version-one worker messages, mirrored by Rust's backend boundary."""

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


Identifier = Annotated[int, Field(ge=0, le=2**64 - 1)]
TokenId = Annotated[int, Field(ge=0, le=2**32 - 1)]
SessionId = Annotated[str, Field(min_length=1, max_length=256)]


class AcceptedToken(Record):
    turn_id: Identifier
    index: Identifier
    token_id: TokenId


class Open(Record):
    type: Literal["open"] = "open"
    operation_id: Identifier
    session_id: SessionId


class Prefill(Record):
    type: Literal["prefill"] = "prefill"
    operation_id: Identifier
    session_id: SessionId
    turn_id: Identifier
    generation: Identifier
    audio_offset: Annotated[int, Field(ge=0)]
    audio_bytes: Annotated[int, Field(gt=0)]
    accepted: AcceptedToken | None = None


class Prepare(Record):
    type: Literal["prepare"] = "prepare"
    operation_id: Identifier
    session_id: SessionId
    turn_id: Identifier
    generation: Identifier
    audio_offset: Annotated[int, Field(ge=0)]
    audio_bytes: Annotated[int, Field(gt=0)]
    accepted: AcceptedToken | None = None


class Activate(Record):
    type: Literal["activate"] = "activate"
    operation_id: Identifier
    session_id: SessionId
    turn_id: Identifier
    generation: Identifier


class DiscardPrepared(Record):
    type: Literal["discard_prepared"] = "discard_prepared"
    operation_id: Identifier
    session_id: SessionId


class Decode(Record):
    type: Literal["decode"] = "decode"
    operation_id: Identifier
    session_id: SessionId
    turn_id: Identifier
    generation: Identifier
    accepted: AcceptedToken


class Close(Record):
    type: Literal["close"] = "close"
    operation_id: Identifier
    session_id: SessionId


Operation = Annotated[
    Open | Prefill | Prepare | Activate | DiscardPrepared | Decode | Close,
    Field(discriminator="type"),
]


class Request(Record):
    request_id: Identifier
    body_bytes: Annotated[int, Field(ge=0)]
    operations: Annotated[tuple[Operation, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_batch(self) -> "Request":
        first_type = type(self.operations[0])
        if any(type(operation) is not first_type for operation in self.operations):
            raise ValueError("A worker batch must contain one operation kind")
        if len({operation.session_id for operation in self.operations}) != len(self.operations):
            raise ValueError("A session may occur only once in a worker batch")
        if len({operation.operation_id for operation in self.operations}) != len(self.operations):
            raise ValueError("Operation identifiers must be unique within a batch")
        expected_offset = 0
        for operation in self.operations:
            match operation:
                case (
                    Prefill(audio_offset=offset, audio_bytes=length)
                    | Prepare(audio_offset=offset, audio_bytes=length)
                ):
                    if offset != expected_offset or length % 2:
                        raise ValueError("PCM16 slices must be contiguous, ordered and even-sized")
                    expected_offset += length
        if expected_offset != self.body_bytes:
            raise ValueError("Audio slices must cover the binary body exactly")
        return self


class Opened(Record):
    type: Literal["opened"] = "opened"


class Closed(Record):
    type: Literal["closed"] = "closed"


class Discarded(Record):
    type: Literal["discarded"] = "discarded"


class Token(Record):
    type: Literal["token"] = "token"
    token_id: TokenId
    text_delta: str
    eos: bool
    context_tokens: Annotated[int, Field(ge=0)]


class ErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    SESSION_EXISTS = "session_exists"
    SESSION_NOT_FOUND = "session_not_found"
    TURN_NOT_FOUND = "turn_not_found"
    INVALID_STATE = "invalid_state"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    CONTEXT_LIMIT = "context_limit"
    HISTORY_LIMIT = "history_limit"
    CHANNEL_SATURATED = "channel_saturated"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    BACKEND_FAILED = "backend_failed"
    SLOW_CONSUMER = "slow_consumer"
    SHUTDOWN = "shutdown"


class Failed(Record):
    type: Literal["failed"] = "failed"
    code: ErrorCode
    message: str


Outcome = Annotated[Opened | Closed | Discarded | Token | Failed, Field(discriminator="type")]


class OperationResult(Record):
    operation_id: Identifier
    session_id: SessionId
    turn_id: Identifier | None = None
    generation: Identifier | None = None
    outcome: Outcome


class Timing(Record):
    elapsed_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    encode_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    prefill_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    decode_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0


class Memory(Record):
    allocated_bytes: Identifier = 0
    reserved_bytes: Identifier = 0


class Response(Record):
    request_id: Identifier
    body_bytes: Literal[0] = 0
    results: tuple[OperationResult, ...]
    timing: Timing
    memory: Memory


class Ready(Record):
    type: Literal["ready"] = "ready"
    body_bytes: Literal[0] = 0
    protocol_version: Literal[1] = 1
    model_id: str
    max_context_tokens: Annotated[int, Field(gt=0)]
    max_batch_size: Annotated[int, Field(gt=0)]
    max_audio_samples: Annotated[int, Field(gt=0)]

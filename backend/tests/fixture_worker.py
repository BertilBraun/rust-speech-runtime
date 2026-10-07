"""Explicit network benchmark fixture; never imported by the production executable."""

import argparse
import asyncio
import time
from dataclasses import dataclass, replace

from voice_worker.engine import Engine
from voice_worker.framing import FrameLimits
from voice_worker.protocol import (
    AcceptedToken,
    Activate,
    Close,
    Closed,
    Decode,
    Discarded,
    DiscardPrepared,
    ErrorCode,
    Failed,
    Memory,
    Open,
    Opened,
    OperationResult,
    Prefill,
    Prepare,
    Ready,
    Request,
    Response,
    Timing,
    Token,
)
from voice_worker.server import WorkerServer


@dataclass(frozen=True)
class FixtureConfig:
    response_tokens: int = 12
    prefill_ms: float = 20.0
    decode_ms: float = 12.0
    max_sessions: int = 256
    max_batch_size: int = 16
    max_context_tokens: int = 4096


@dataclass
class FixtureSession:
    context_tokens: int = 0
    proposed: AcceptedToken | None = None
    consumed: AcceptedToken | None = None
    audio_bytes: int = 0


@dataclass(frozen=True)
class FixturePreparation:
    session: FixtureSession
    turn_id: int
    generation: int
    token: Token


class FixtureEngine(Engine):
    def __init__(self, configuration: FixtureConfig) -> None:
        self.configuration = configuration
        self.sessions: dict[str, FixtureSession] = {}
        self.prepared_sessions: dict[str, FixturePreparation] = {}

    def ready(self) -> Ready:
        return Ready(
            model_id="benchmark-fixture-v1",
            max_context_tokens=self.configuration.max_context_tokens,
            max_batch_size=self.configuration.max_batch_size,
            max_audio_samples=480_000,
        )

    def reset(self) -> None:
        self.prepared_sessions.clear()
        self.sessions.clear()

    def execute(self, request: Request, body: bytes) -> Response:
        started = time.perf_counter()
        if len(request.operations) > self.configuration.max_batch_size:
            raise ValueError("Fixture batch size exceeded")
        if len(body) != request.body_bytes:
            raise ValueError("Binary body size mismatch")
        encode_ms = 0.0
        prefill_ms = 0.0
        decode_ms = 0.0
        match request.operations[0]:
            case Prefill() | Prepare():
                time.sleep(self.configuration.prefill_ms / 1000)
                prefill_ms = (time.perf_counter() - started) * 1000
            case Decode():
                time.sleep(self.configuration.decode_ms / 1000)
                decode_ms = (time.perf_counter() - started) * 1000
        results = []
        for operation in request.operations:
            turn_id = None
            generation = None
            match operation:
                case Open(session_id=identifier):
                    if identifier in self.sessions:
                        outcome = Failed(
                            code=ErrorCode.SESSION_EXISTS, message="Session already exists"
                        )
                    elif len(self.sessions) >= self.configuration.max_sessions:
                        outcome = Failed(
                            code=ErrorCode.CAPACITY_EXCEEDED, message="Fixture cache full"
                        )
                    else:
                        self.sessions[identifier] = FixtureSession()
                        outcome = Opened()
                case Close(session_id=identifier):
                    self.prepared_sessions.pop(identifier, None)
                    self.sessions.pop(identifier, None)
                    outcome = Closed()
                case DiscardPrepared(session_id=identifier):
                    self.prepared_sessions.pop(identifier, None)
                    outcome = Discarded()
                case Activate(session_id=identifier, turn_id=turn_id, generation=generation):
                    prepared = self.prepared_sessions.get(identifier)
                    if identifier not in self.sessions:
                        outcome = Failed(
                            code=ErrorCode.SESSION_NOT_FOUND, message="Unknown cache handle"
                        )
                    elif (
                        prepared is None
                        or prepared.turn_id != turn_id
                        or prepared.generation != generation
                    ):
                        outcome = Failed(
                            code=ErrorCode.INVALID_STATE, message="No matching prepared turn"
                        )
                    else:
                        self.sessions[identifier] = prepared.session
                        self.prepared_sessions.pop(identifier)
                        outcome = prepared.token
                case Prefill(session_id=identifier) | Prepare(session_id=identifier):
                    self.prepared_sessions.pop(identifier, None)
                    turn_id = operation.turn_id
                    generation = operation.generation
                    session = self.sessions.get(identifier)
                    if session is None:
                        outcome = Failed(
                            code=ErrorCode.SESSION_NOT_FOUND, message="Unknown cache handle"
                        )
                    elif operation.audio_bytes > 960_000:
                        outcome = Failed(code=ErrorCode.INVALID_INPUT, message="Oversize utterance")
                    elif operation.accepted is not None and operation.accepted not in (
                        session.proposed,
                        session.consumed,
                    ):
                        outcome = Failed(
                            code=ErrorCode.INVALID_INPUT, message="Invalid accepted token"
                        )
                    else:
                        match operation:
                            case Prepare():
                                session = replace(session)
                        if (
                            operation.accepted is not None
                            and operation.accepted != session.consumed
                        ):
                            session.context_tokens += 1
                        session.audio_bytes += operation.audio_bytes
                        session.context_tokens += 12 + (operation.audio_bytes + 3199) // 3200
                        session.consumed = None
                        outcome = self._propose(session, turn_id, 0)
                        match operation, outcome:
                            case Prepare(), Token():
                                self.prepared_sessions[identifier] = FixturePreparation(
                                    session, turn_id, generation, outcome
                                )
                case Decode(session_id=identifier, accepted=accepted):
                    self.prepared_sessions.pop(identifier, None)
                    turn_id = operation.turn_id
                    generation = operation.generation
                    session = self.sessions.get(identifier)
                    if session is None:
                        outcome = Failed(
                            code=ErrorCode.SESSION_NOT_FOUND, message="Unknown cache handle"
                        )
                    elif accepted != session.proposed or accepted.turn_id != turn_id:
                        outcome = Failed(
                            code=ErrorCode.INVALID_INPUT, message="Invalid accepted token"
                        )
                    else:
                        session.consumed = accepted
                        session.context_tokens += 1
                        outcome = self._propose(session, turn_id, accepted.index + 1)
            results.append(
                OperationResult(
                    operation_id=operation.operation_id,
                    session_id=operation.session_id,
                    turn_id=turn_id,
                    generation=generation,
                    outcome=outcome,
                )
            )
        return Response(
            request_id=request.request_id,
            results=tuple(results),
            timing=Timing(
                elapsed_ms=(time.perf_counter() - started) * 1000,
                encode_ms=encode_ms,
                prefill_ms=prefill_ms,
                decode_ms=decode_ms,
            ),
            memory=Memory(
                allocated_bytes=sum(
                    session.context_tokens * 12288
                    for session in (
                        *self.sessions.values(),
                        *(prepared.session for prepared in self.prepared_sessions.values()),
                    )
                )
            ),
        )

    def _propose(self, session: FixtureSession, turn_id: int, index: int) -> Token | Failed:
        if session.context_tokens >= self.configuration.max_context_tokens:
            return Failed(
                code=ErrorCode.CONTEXT_LIMIT, message="Fixture conversation context is full"
            )
        eos = index >= self.configuration.response_tokens
        token_id = 248046 if eos else 1000 + index
        session.proposed = AcceptedToken(turn_id=turn_id, index=index, token_id=token_id)
        return Token(
            token_id=token_id,
            text_delta="" if eos else f" token{index}",
            eos=eos,
            context_tokens=session.context_tokens,
        )


async def run(port: int, configuration: FixtureConfig, exit_on_disconnect: bool = False) -> None:
    worker = WorkerServer(
        FixtureEngine(configuration),
        FrameLimits(1024 * 1024, configuration.max_batch_size * 960000, 60.0),
    )
    disconnected = asyncio.Event()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        owns_connection = not worker.connected
        try:
            await worker.handle(reader, writer)
        finally:
            if owns_connection:
                disconnected.set()

    server = await asyncio.start_server(handle, "127.0.0.1", port)
    bound_port = server.sockets[0].getsockname()[1]
    print(f"Fixture worker ready on 127.0.0.1:{bound_port}", flush=True)
    try:
        async with server:
            if exit_on_disconnect:
                await disconnected.wait()
            else:
                await server.serve_forever()
    finally:
        await worker.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic benchmark worker, not a model")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--response-tokens", type=int, default=12)
    parser.add_argument("--prefill-ms", type=float, default=20.0)
    parser.add_argument("--decode-ms", type=float, default=12.0)
    parser.add_argument("--max-sessions", type=int, default=256)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--exit-on-disconnect", action="store_true")
    arguments = parser.parse_args()
    configuration = FixtureConfig(
        arguments.response_tokens,
        arguments.prefill_ms,
        arguments.decode_ms,
        arguments.max_sessions,
        arguments.max_batch_size,
    )
    if (
        min(configuration.response_tokens, configuration.max_sessions, configuration.max_batch_size)
        < 1
    ):
        parser.error("Token/session/batch limits must be positive")
    if min(configuration.prefill_ms, configuration.decode_ms) < 0:
        parser.error("Latencies must be nonnegative")
    asyncio.run(run(arguments.port, configuration, arguments.exit_on_disconnect))


if __name__ == "__main__":
    main()

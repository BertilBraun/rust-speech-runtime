"""Resident model/cache owner executing bounded batches chosen by Rust."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor
from transformers.cache_utils import DynamicCache

from voice_worker.cache import batch_workspace_bytes, cache_bytes, reservation_bytes
from voice_worker.config import WorkerConfig
from voice_worker.detokenizer import preview_text
from voice_worker.engine import Engine
from voice_worker.model import ASSISTANT_SUFFIX, EOS, USER_PREFIX, SpeechModel, pseudo_token_count
from voice_worker.protocol import (
    AcceptedToken,
    Close,
    Closed,
    Decode,
    ErrorCode,
    Failed,
    Memory,
    Open,
    Opened,
    Operation,
    OperationResult,
    Prefill,
    Ready,
    Request,
    Response,
    Timing,
    Token,
)
from voice_worker.session import ModelSession, Proposal

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedPrefill:
    operation: Prefill
    session: ModelSession
    audio: bytes
    prefix: tuple[int, ...]


@dataclass(frozen=True)
class PreparedForward:
    operation: Prefill | Decode
    session: ModelSession
    embeddings: Tensor


def result_for(operation: Operation, outcome: Opened | Closed | Token | Failed) -> OperationResult:
    match operation:
        case (
            Prefill(turn_id=turn_id, generation=generation)
            | Decode(turn_id=turn_id, generation=generation)
        ):
            return OperationResult(
                operation_id=operation.operation_id,
                session_id=operation.session_id,
                turn_id=turn_id,
                generation=generation,
                outcome=outcome,
            )
        case Open() | Close():
            return OperationResult(
                operation_id=operation.operation_id,
                session_id=operation.session_id,
                outcome=outcome,
            )


class PyTorchEngine(Engine):
    def __init__(self, configuration: WorkerConfig, model: SpeechModel) -> None:
        self.configuration = configuration
        self.model = model
        self.sessions: dict[str, ModelSession] = {}
        self.reservation = reservation_bytes(model.text_config, configuration.max_context_tokens)
        if self.reservation > configuration.cache_budget_bytes:
            raise ValueError("Cache budget cannot reserve even one complete session")

    def ready(self) -> Ready:
        return Ready(
            model_id=self.model.model_id,
            max_context_tokens=self.configuration.max_context_tokens,
            max_batch_size=self.configuration.max_batch_size,
            max_audio_samples=480_000,
        )

    def reset(self) -> None:
        self.sessions.clear()

    def execute(self, request: Request, body: bytes) -> Response:
        started = time.perf_counter()
        if (
            len(request.operations) > self.configuration.max_batch_size
            or len(body) != request.body_bytes
        ):
            raise ValueError("Batch shape or binary body differs from worker capabilities")
        with torch.inference_mode():
            match request.operations[0]:
                case Open():
                    results = tuple(
                        self._open(cast(Open, operation)) for operation in request.operations
                    )
                    timing = Timing()
                case Close():
                    results = tuple(
                        self._close(cast(Close, operation)) for operation in request.operations
                    )
                    timing = Timing()
                case Prefill():
                    results, timing = self._prefill(
                        cast(tuple[Prefill, ...], request.operations), body
                    )
                case Decode():
                    results, timing = self._decode(cast(tuple[Decode, ...], request.operations))
        return Response(
            request_id=request.request_id,
            results=self._ordered_results(request, results),
            timing=Timing(
                elapsed_ms=(time.perf_counter() - started) * 1000,
                encode_ms=timing.encode_ms,
                prefill_ms=timing.prefill_ms,
                decode_ms=timing.decode_ms,
            ),
            memory=Memory(
                allocated_bytes=torch.cuda.memory_allocated(self.model.device),
                reserved_bytes=torch.cuda.memory_reserved(self.model.device),
            ),
        )

    @staticmethod
    def _ordered_results(
        request: Request, results: tuple[OperationResult, ...]
    ) -> tuple[OperationResult, ...]:
        by_identifier = {result.operation_id: result for result in results}
        assert len(by_identifier) == len(results) == len(request.operations)
        return tuple(by_identifier[operation.operation_id] for operation in request.operations)

    def _open(self, operation: Open) -> OperationResult:
        if operation.session_id in self.sessions:
            return result_for(
                operation, Failed(code=ErrorCode.SESSION_EXISTS, message="Session already exists")
            )
        reserved = (len(self.sessions) + 1) * self.reservation
        free, _ = torch.cuda.mem_get_info(self.model.device)
        if (
            len(self.sessions) >= self.configuration.max_sessions
            or reserved > self.configuration.cache_budget_bytes
            or free
            < self.configuration.workspace_reserve_bytes
            + self._unmaterialized_reservations()
            + self.reservation
        ):
            return result_for(
                operation,
                Failed(code=ErrorCode.CAPACITY_EXCEEDED, message="Cache capacity exhausted"),
            )
        self.sessions[operation.session_id] = ModelSession(
            DynamicCache(config=self.model.text_config)
        )
        return result_for(operation, Opened())

    def _unmaterialized_reservations(self) -> int:
        return sum(
            max(0, self.reservation - cache_bytes(session.cache, self.model.text_config))
            for session in self.sessions.values()
        )

    def _close(self, operation: Close) -> OperationResult:
        self.sessions.pop(operation.session_id, None)
        return result_for(operation, Closed())

    def _prefill(
        self, operations: tuple[Prefill, ...], body: bytes
    ) -> tuple[tuple[OperationResult, ...], Timing]:
        prepared = []
        results = []
        for operation in operations:
            session = self.sessions.get(operation.session_id)
            if session is None:
                results.append(
                    result_for(
                        operation,
                        Failed(code=ErrorCode.SESSION_NOT_FOUND, message="No cache handle"),
                    )
                )
                continue
            audio = body[operation.audio_offset : operation.audio_offset + operation.audio_bytes]
            try:
                if operation.audio_bytes > 960_000:
                    raise ValueError("Utterance exceeds thirty seconds")
                if session.generation is not None and operation.generation <= session.generation:
                    raise ValueError("Generation identifiers must increase within a session")
                prefix = session.next_turn_prefix(operation.accepted) + USER_PREFIX
                appended = len(prefix) + pseudo_token_count(len(audio) // 2) + len(ASSISTANT_SUFFIX)
                if (
                    session.cache.get_seq_length() + appended
                    >= self.configuration.max_context_tokens
                ):
                    results.append(
                        result_for(
                            operation,
                            Failed(
                                code=ErrorCode.CONTEXT_LIMIT, message="Conversation context is full"
                            ),
                        )
                    )
                    continue
                prepared.append(PreparedPrefill(operation, session, audio, prefix))
            except ValueError as error:
                results.append(
                    result_for(operation, Failed(code=ErrorCode.INVALID_INPUT, message=str(error)))
                )
        if not prepared:
            return tuple(results), Timing()
        try:
            projected, encoder_timer = self.model.encode_audio(
                tuple(item.audio for item in prepared)
            )
            forwards = tuple(
                PreparedForward(
                    item.operation,
                    item.session,
                    torch.cat(
                        (self.model.embed(item.prefix), speech, self.model.embed(ASSISTANT_SUFFIX))
                    ),
                )
                for item, speech in zip(prepared, projected, strict=True)
            )
            completed, forward_ms = self._forward_groups(forwards)
            results.extend(completed)
            return tuple(results), Timing(
                encode_ms=encoder_timer.elapsed_ms(), prefill_ms=forward_ms
            )
        except (RuntimeError, ValueError) as error:
            LOGGER.exception("Prefill failed; affected caches invalidated")
            for item in prepared:
                self.sessions.pop(item.operation.session_id, None)
                results.append(
                    result_for(
                        item.operation, Failed(code=ErrorCode.BACKEND_FAILED, message=str(error))
                    )
                )
            return tuple(results), Timing()

    def _decode(self, operations: tuple[Decode, ...]) -> tuple[tuple[OperationResult, ...], Timing]:
        forwards = []
        results = []
        for operation in operations:
            session = self.sessions.get(operation.session_id)
            if session is None:
                results.append(
                    result_for(
                        operation,
                        Failed(code=ErrorCode.SESSION_NOT_FOUND, message="No cache handle"),
                    )
                )
                continue
            try:
                if (
                    operation.turn_id != session.turn_id
                    or operation.generation != session.generation
                ):
                    raise ValueError("Decode epoch differs from the active turn")
                session.validate_acceptance(operation.accepted)
                if operation.accepted.token_id == EOS:
                    raise ValueError("EOS ends the turn and cannot request another proposal")
                if session.cache.get_seq_length() + 1 >= self.configuration.max_context_tokens:
                    results.append(
                        result_for(
                            operation,
                            Failed(
                                code=ErrorCode.CONTEXT_LIMIT, message="Conversation context is full"
                            ),
                        )
                    )
                    continue
                forwards.append(
                    PreparedForward(
                        operation, session, self.model.embed((operation.accepted.token_id,))
                    )
                )
            except ValueError as error:
                results.append(
                    result_for(operation, Failed(code=ErrorCode.INVALID_INPUT, message=str(error)))
                )
        if not forwards:
            return tuple(results), Timing()
        try:
            completed, elapsed = self._forward_groups(tuple(forwards))
            results.extend(completed)
            return tuple(results), Timing(decode_ms=elapsed)
        except (RuntimeError, ValueError) as error:
            LOGGER.exception("Decode failed; affected caches invalidated")
            for item in forwards:
                self.sessions.pop(item.operation.session_id, None)
                results.append(
                    result_for(
                        item.operation, Failed(code=ErrorCode.BACKEND_FAILED, message=str(error))
                    )
                )
            return tuple(results), Timing()

    def _forward_groups(
        self, forwards: tuple[PreparedForward, ...]
    ) -> tuple[tuple[OperationResult, ...], float]:
        groups: dict[tuple[int, bool], list[PreparedForward]] = defaultdict(list)
        for item in forwards:
            key = (item.embeddings.shape[0], item.session.cache.get_seq_length() > 0)
            groups[key].append(item)
        results = []
        elapsed = 0.0
        for group in groups.values():
            lengths = tuple(item.session.cache.get_seq_length() for item in group)
            workspace = batch_workspace_bytes(
                self.model.text_config, lengths, group[0].embeddings.shape[0]
            )
            free, _ = torch.cuda.mem_get_info(self.model.device)
            if (
                free
                < self.configuration.workspace_reserve_bytes
                + self._unmaterialized_reservations()
                + workspace
            ):
                results.extend(
                    result_for(
                        item.operation,
                        Failed(
                            code=ErrorCode.CAPACITY_EXCEEDED, message="Batch workspace unavailable"
                        ),
                    )
                    for item in group
                )
                continue
            completed = self.model.forward_batch(
                tuple(item.embeddings for item in group),
                tuple(item.session.cache for item in group),
            )
            elapsed += completed.timer.elapsed_ms()
            for item, cache, token in zip(
                group, completed.caches, completed.token_ids, strict=True
            ):
                session = item.session
                session.cache = cache
                match item.operation:
                    case Prefill(turn_id=turn_id, generation=generation):
                        session.start_turn(turn_id, generation)
                        index = 0
                    case Decode(accepted=accepted):
                        session.accept(accepted)
                        index = accepted.index + 1
                proposal = AcceptedToken(
                    turn_id=item.operation.turn_id, index=index, token_id=token
                )
                text = preview_text(
                    session.pending_text, self.model.token_piece(token), token == EOS
                )
                session.proposed = Proposal(proposal, text)
                results.append(
                    result_for(
                        item.operation,
                        Token(
                            token_id=token,
                            text_delta=text.delta,
                            eos=token == EOS,
                            context_tokens=session.cache.get_seq_length(),
                        ),
                    )
                )
        return tuple(results), elapsed

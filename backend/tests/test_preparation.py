from collections.abc import Sequence
from typing import Literal

import pytest
import torch
from test_engine import engine as engine
from test_engine import execute, prefill
from test_engine import tiny_model as tiny_model
from torch import Tensor
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from voice_worker.cache import attention_layer, cache_bytes, join_caches, recurrent_layer
from voice_worker.config import WarmupConfig, WorkerConfig
from voice_worker.model import ForwardBatch, StageTimer
from voice_worker.protocol import (
    AcceptedToken,
    Activate,
    Close,
    Decode,
    Discarded,
    DiscardPrepared,
    ErrorCode,
    Failed,
    Open,
    Prepare,
    Token,
)
from voice_worker.pytorch_engine import PyTorchEngine


def prepare(generation: int, accepted: AcceptedToken | None = None) -> Prepare:
    return Prepare(
        operation_id=2,
        session_id="a",
        turn_id=generation,
        generation=generation,
        audio_offset=0,
        audio_bytes=3200,
        accepted=accepted,
    )


def activate(generation: int) -> Activate:
    return Activate(operation_id=3, session_id="a", turn_id=generation, generation=generation)


def assert_cache_identical(
    first: DynamicCache, second: DynamicCache, configuration: Qwen3_5TextConfig
) -> None:
    assert first.get_seq_length() == second.get_seq_length()
    for index, layer_type in enumerate(configuration.layer_types):
        match layer_type:
            case "full_attention":
                first_layer = attention_layer(first, index)
                second_layer = attention_layer(second, index)
                tensors = (
                    (first_layer.keys, second_layer.keys),
                    (first_layer.values, second_layer.values),
                )
            case "linear_attention":
                first_linear = recurrent_layer(first, index)
                second_linear = recurrent_layer(second, index)
                tensors = (
                    (first_linear.conv_states, second_linear.conv_states),
                    (first_linear.recurrent_states, second_linear.recurrent_states),
                )
            case _:
                raise AssertionError(layer_type)
        for original, snapshot in tensors:
            torch.testing.assert_close(original, snapshot, atol=0, rtol=0)


def test_prepare_and_activate_preserve_multiturn_history_exactly(engine: PyTorchEngine) -> None:
    execute(engine, (Open(operation_id=0, session_id="a"), Open(operation_id=1, session_id="b")))
    accepted = None
    for generation in (1, 2, 3):
        canonical = engine.sessions["a"]
        snapshot = join_caches((canonical.cache,), engine.model.text_config)
        prepared = execute(engine, (prepare(generation, accepted),), bytes(3200)).results[0]
        assert isinstance(prepared.outcome, Token)
        assert engine.sessions["a"] is canonical
        assert_cache_identical(canonical.cache, snapshot, engine.model.text_config)
        expected_generation = generation - 1 if generation > 1 else None
        assert canonical.generation == expected_generation
        committed = execute(engine, (activate(generation),)).results[0]
        normal = execute(
            engine,
            (prefill("b", 4, generation, generation, 0, 3200, accepted),),
            bytes(3200),
        ).results[0]
        assert committed.outcome == prepared.outcome == normal.outcome
        assert_cache_identical(
            engine.sessions["a"].cache, engine.sessions["b"].cache, engine.model.text_config
        )
        assert not engine.prepared_sessions
        accepted = AcceptedToken(turn_id=generation, index=0, token_id=prepared.outcome.token_id)


def test_activation_fences_epoch_and_reuses_first_token_without_forward(
    engine: PyTorchEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    execute(engine, (Open(operation_id=0, session_id="a"),))
    prepared = execute(engine, (prepare(1),), bytes(3200)).results[0]

    def unexpected_encode(audio: Sequence[bytes]) -> tuple[tuple[Tensor, ...], StageTimer]:
        raise AssertionError("Activation must not re-encode audio")

    monkeypatch.setattr(engine.model, "encode_audio", unexpected_encode)
    invalid = execute(engine, (activate(2),)).results[0].outcome
    assert isinstance(invalid, Failed)
    assert invalid.code is ErrorCode.INVALID_STATE
    assert engine.sessions["a"].cache.get_seq_length() == 0
    committed = execute(engine, (activate(1),)).results[0].outcome
    assert committed == prepared.outcome
    assert isinstance(committed, Token)
    decoded = (
        execute(
            engine,
            (
                Decode(
                    operation_id=4,
                    session_id="a",
                    turn_id=1,
                    generation=1,
                    accepted=AcceptedToken(turn_id=1, index=0, token_id=committed.token_id),
                ),
            ),
        )
        .results[0]
        .outcome
    )
    assert isinstance(decoded, Token)


def test_discard_and_resumed_audio_retry_only_retains_new_snapshot(engine: PyTorchEngine) -> None:
    execute(engine, (Open(operation_id=0, session_id="a"),))
    execute(engine, (prepare(1),), bytes(3200))
    discarded = execute(engine, (DiscardPrepared(operation_id=4, session_id="a"),)).results[0]
    assert isinstance(discarded.outcome, Discarded)
    assert not engine.prepared_sessions
    assert engine.sessions["a"].cache.get_seq_length() == 0
    assert isinstance(execute(engine, (activate(1),)).results[0].outcome, Failed)
    newer = prepare(2).model_copy(update={"audio_bytes": 6400})
    result = execute(engine, (newer,), bytes(6400)).results[0].outcome
    assert execute(engine, (activate(2),)).results[0].outcome == result
    assert engine.sessions["a"].cache.get_seq_length() == 14


@pytest.mark.parametrize("failure_stage", ["encode", "forward"])
def test_failed_preparation_does_not_invalidate_committed_cache(
    engine: PyTorchEngine,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: Literal["encode", "forward"],
) -> None:
    execute(engine, (Open(operation_id=0, session_id="a"),))
    execute(engine, (prefill("a", 1, 1, 1, 0, 3200),), bytes(3200))
    canonical = engine.sessions["a"]
    snapshot = join_caches((canonical.cache,), engine.model.text_config)

    def failed_encode(audio: Sequence[bytes]) -> tuple[tuple[Tensor, ...], StageTimer]:
        raise RuntimeError("Test allocation failure")

    original_forward = engine.model.forward_batch

    def failed_forward(
        embeddings: Sequence[Tensor], caches: Sequence[DynamicCache]
    ) -> ForwardBatch:
        original_forward(embeddings, caches)
        raise RuntimeError("Test failure after completed forward")

    match failure_stage:
        case "encode":
            monkeypatch.setattr(engine.model, "encode_audio", failed_encode)
        case "forward":
            monkeypatch.setattr(engine.model, "forward_batch", failed_forward)
    result = execute(engine, (prepare(2),), bytes(3200)).results[0].outcome
    assert isinstance(result, Failed)
    assert result.code is ErrorCode.BACKEND_FAILED
    assert engine.sessions["a"] is canonical
    assert_cache_identical(canonical.cache, snapshot, engine.model.text_config)
    assert not engine.prepared_sessions


def test_optional_cache_reservation_never_steals_committed_session_capacity(
    engine: PyTorchEngine,
) -> None:
    engine.configuration = WorkerConfig(
        model=engine.configuration.model,
        device="cuda:0",
        cache_budget_bytes=engine.reservation * 2,
    )
    execute(engine, (Open(operation_id=0, session_id="a"),))
    result = execute(engine, (prepare(1),), bytes(3200)).results[0].outcome
    assert isinstance(result, Token)
    actual = cache_bytes(engine.prepared_sessions["a"].session.cache, engine.model.text_config)
    assert engine._unmaterialized_reservations() == 2 * engine.reservation - actual
    rejected = execute(engine, (Open(operation_id=4, session_id="b"),)).results[0].outcome
    assert isinstance(rejected, Failed)
    assert rejected.code is ErrorCode.CAPACITY_EXCEEDED
    execute(engine, (DiscardPrepared(operation_id=5, session_id="a"),))
    execute(engine, (Open(operation_id=6, session_id="b"),))
    rejected = execute(engine, (prepare(1),), bytes(3200)).results[0].outcome
    assert isinstance(rejected, Failed)
    assert rejected.code is ErrorCode.CAPACITY_EXCEEDED
    fallback = execute(engine, (prefill("a", 7, 1, 1, 0, 3200),), bytes(3200)).results[0].outcome
    assert isinstance(fallback, Token)


@pytest.mark.parametrize("close_session", [True, False])
def test_close_and_reset_free_preparation(engine: PyTorchEngine, close_session: bool) -> None:
    execute(engine, (Open(operation_id=0, session_id="a"),))
    execute(engine, (prepare(1),), bytes(3200))
    if close_session:
        execute(engine, (Close(operation_id=3, session_id="a"),))
    else:
        engine.reset()
    assert not engine.prepared_sessions
    assert not engine.sessions


def test_normal_prefill_invalidates_preparation(engine: PyTorchEngine) -> None:
    execute(engine, (Open(operation_id=0, session_id="a"),))
    execute(engine, (prepare(1),), bytes(3200))
    execute(engine, (prefill("a", 4, 2, 2, 0, 6400),), bytes(6400))
    assert not engine.prepared_sessions
    assert engine.sessions["a"].turn_id == 2
    assert isinstance(execute(engine, (activate(1),)).results[0].outcome, Failed)


def test_decode_invalidates_preparation_before_mutating_canonical_history(
    engine: PyTorchEngine,
) -> None:
    execute(engine, (Open(operation_id=0, session_id="a"),))
    first = execute(engine, (prefill("a", 1, 1, 1, 0, 3200),), bytes(3200)).results[0].outcome
    assert isinstance(first, Token)
    accepted = AcceptedToken(turn_id=1, index=0, token_id=first.token_id)
    execute(engine, (prepare(2, accepted),), bytes(3200))
    decoded = (
        execute(
            engine,
            (Decode(operation_id=4, session_id="a", turn_id=1, generation=1, accepted=accepted),),
        )
        .results[0]
        .outcome
    )
    assert isinstance(decoded, Token)
    assert not engine.prepared_sessions
    assert engine.sessions["a"].generation == 1
    assert isinstance(execute(engine, (activate(2),)).results[0].outcome, Failed)


@pytest.mark.parametrize("worker_batch_size,prefill_batch_size", [(1, 4), (2, 4), (4, 3), (4, 4)])
def test_warmup_covers_populated_prefill_and_bounded_decode_batches(
    engine: PyTorchEngine,
    monkeypatch: pytest.MonkeyPatch,
    worker_batch_size: int,
    prefill_batch_size: int,
) -> None:
    model = engine.model
    model.configuration = engine.configuration.model_copy(
        update={
            "max_batch_size": worker_batch_size,
            "max_context_tokens": 512,
            "warmup": WarmupConfig(max_prefill_batch_size=prefill_batch_size),
        }
    )
    original_forward = model.forward_batch
    original_encode = model.encode_audio
    calls: list[tuple[int, int, int]] = []
    encoder_sizes: list[int] = []

    def tracked_forward(
        embeddings: Sequence[Tensor], caches: Sequence[DynamicCache]
    ) -> ForwardBatch:
        assert len(embeddings) == len(caches)
        calls.append((len(embeddings), embeddings[0].shape[0], caches[0].get_seq_length()))
        return original_forward(embeddings, caches)

    def tracked_encode(audio: Sequence[bytes]) -> tuple[tuple[Tensor, ...], StageTimer]:
        encoder_sizes.append(len(audio))
        return original_encode(audio)

    monkeypatch.setattr(model, "forward_batch", tracked_forward)
    monkeypatch.setattr(model, "encode_audio", tracked_encode)
    model.warmup()
    maximum_prefill = min(worker_batch_size, prefill_batch_size)
    assert encoder_sizes == list(range(1, maximum_prefill + 1))
    assert max(batch for batch, _, _ in calls) == worker_batch_size
    assert max(batch for batch, length, _ in calls if length > 1) == maximum_prefill
    assert all(length + previous < 512 for _, length, previous in calls)
    for batch in range(1, maximum_prefill + 1):
        assert (batch, 64, 0) in calls
        assert (batch, 66, 64) in calls
        assert (batch, 67, 64) in calls
    assert any(
        batch == worker_batch_size and length == 1 and previous > 0
        for batch, length, previous in calls
    )
    assert not engine.sessions

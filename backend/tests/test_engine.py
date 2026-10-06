from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from test_hybrid_cache import assert_cache_equal
from test_hybrid_cache import tiny_model as tiny_model
from torch import Tensor
from transformers import Qwen3_5ForCausalLM

from voice_worker.cache import batch_workspace_bytes, cache_bytes
from voice_worker.config import ModelManifest, WorkerConfig
from voice_worker.model import SpeechModel, StageTimer, pseudo_token_count
from voice_worker.protocol import (
    AcceptedToken,
    Decode,
    Failed,
    Open,
    Opened,
    Operation,
    Prefill,
    Request,
    Response,
    Token,
)
from voice_worker.pytorch_engine import PyTorchEngine


class FixtureSpeechModel(SpeechModel):
    def __init__(self, language_model: Qwen3_5ForCausalLM) -> None:
        self.device = torch.device("cpu")
        self.language_model = language_model
        self.text_config = language_model.config
        self.model_id = "tiny-random-qwen-cpu-fixture"

    def embed(self, token_ids: Sequence[int]) -> Tensor:
        indices = torch.tensor(
            [identifier % self.text_config.vocab_size for identifier in token_ids]
        )
        return self.language_model.get_input_embeddings()(indices)

    def token_piece(self, token_id: int) -> bytes:
        return f"t{token_id}".encode()

    def encode_audio(self, audio: Sequence[bytes]) -> tuple[tuple[Tensor, ...], StageTimer]:
        timer = StageTimer(self.device)
        timer.start()
        embedded = tuple(
            self.embed((9,) * pseudo_token_count(len(packet) // 2)) for packet in audio
        )
        timer.finish()
        return embedded, timer


def memory_information(device: torch.device) -> tuple[int, int]:
    return 2**40, 2**40


def memory_usage(device: torch.device) -> int:
    return 0


@pytest.fixture
def engine(tiny_model: Qwen3_5ForCausalLM, monkeypatch: pytest.MonkeyPatch) -> PyTorchEngine:
    monkeypatch.setattr(torch.cuda, "mem_get_info", memory_information)
    monkeypatch.setattr(torch.cuda, "memory_allocated", memory_usage)
    monkeypatch.setattr(torch.cuda, "memory_reserved", memory_usage)
    configuration = WorkerConfig(
        model=ModelManifest(projector_checkpoint=Path("not-loaded"), projector_sha256="0" * 64),
        device="cuda:0",
    )
    return PyTorchEngine(configuration, FixtureSpeechModel(tiny_model))


def execute(
    engine: PyTorchEngine, operations: tuple[Operation, ...], body: bytes = b""
) -> Response:
    return engine.execute(Request(request_id=1, body_bytes=len(body), operations=operations), body)


def prefill(
    session_id: str,
    operation_id: int,
    turn_id: int,
    generation: int,
    audio_offset: int,
    audio_bytes: int,
    accepted: AcceptedToken | None = None,
) -> Prefill:
    return Prefill(
        operation_id=operation_id,
        session_id=session_id,
        turn_id=turn_id,
        generation=generation,
        audio_offset=audio_offset,
        audio_bytes=audio_bytes,
        accepted=accepted,
    )


def test_grouped_batch_results_keep_input_order_with_validation_failures(
    engine: PyTorchEngine,
) -> None:
    execute(
        engine,
        tuple(
            Open(operation_id=index, session_id=identifier)
            for index, identifier in enumerate(("a", "b", "c"))
        ),
    )
    operations = (
        prefill("a", 20, 1, 1, 0, 3200),
        prefill("missing", 21, 1, 1, 3200, 3200),
        prefill("b", 22, 1, 1, 6400, 6400),
        prefill("c", 23, 1, 1, 12800, 3200),
    )
    response = execute(engine, operations, bytes(16000))
    assert [item.operation_id for item in response.results] == [20, 21, 22, 23]
    assert isinstance(response.results[1].outcome, Failed)
    assert all(isinstance(response.results[index].outcome, Token) for index in (0, 2, 3))


def test_inflight_interruption_matches_final_accepted_token_reconciliation(
    engine: PyTorchEngine,
) -> None:
    execute(engine, (Open(operation_id=0, session_id="a"), Open(operation_id=1, session_id="b")))
    first = execute(
        engine, (prefill("a", 2, 100, 1, 0, 3200), prefill("b", 3, 100, 1, 3200, 3200)), bytes(6400)
    )
    outcome = first.results[0].outcome
    assert isinstance(outcome, Token)
    accepted = AcceptedToken(turn_id=100, index=0, token_id=outcome.token_id)
    execute(
        engine,
        (Decode(operation_id=4, session_id="a", turn_id=100, generation=1, accepted=accepted),),
    )
    second = execute(
        engine,
        (prefill("a", 5, 4, 2, 0, 3200), prefill("b", 6, 4, 2, 3200, 3200, accepted)),
        bytes(6400),
    )
    assert all(isinstance(result.outcome, Token) for result in second.results)
    assert_cache_equal(engine.sessions["a"].cache, engine.sessions["b"].cache)
    assert engine.sessions["a"].cache.get_seq_length() == 29


def test_invalid_accepted_token_and_stale_generation_leave_cache_unchanged(
    engine: PyTorchEngine,
) -> None:
    execute(engine, (Open(operation_id=0, session_id="a"),))
    execute(engine, (prefill("a", 1, 9, 2, 0, 3200),), bytes(3200))
    length = engine.sessions["a"].cache.get_seq_length()
    invalid = execute(
        engine,
        (
            Decode(
                operation_id=2,
                session_id="a",
                turn_id=9,
                generation=2,
                accepted=AcceptedToken(turn_id=9, index=10, token_id=1),
            ),
        ),
    )
    assert isinstance(invalid.results[0].outcome, Failed)
    stale = execute(engine, (prefill("a", 3, 10, 1, 0, 3200),), bytes(3200))
    assert isinstance(stale.results[0].outcome, Failed)
    assert engine.sessions["a"].cache.get_seq_length() == length


def test_open_reserves_unmaterialized_cache_against_actual_free_memory(
    engine: PyTorchEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine.configuration = WorkerConfig(
        model=engine.configuration.model,
        device="cuda:0",
        cache_budget_bytes=engine.reservation * 10,
        workspace_reserve_bytes=1,
    )

    def free_memory(device: torch.device) -> tuple[int, int]:
        return engine.reservation + 1, engine.reservation + 1

    monkeypatch.setattr(torch.cuda, "mem_get_info", free_memory)
    response = execute(
        engine,
        tuple(Open(operation_id=index, session_id=str(index)) for index in range(3)),
    )
    assert isinstance(response.results[0].outcome, Opened)
    assert all(isinstance(result.outcome, Failed) for result in response.results[1:])
    assert len(engine.sessions) == 1


def test_materialized_cache_is_not_charged_twice_against_free_memory(
    engine: PyTorchEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    execute(engine, (Open(operation_id=0, session_id="a"),))
    execute(engine, (prefill("a", 1, 9, 2, 0, 3200),), bytes(3200))
    materialized = cache_bytes(engine.sessions["a"].cache, engine.model.text_config)
    engine.configuration = WorkerConfig(
        model=engine.configuration.model,
        device="cuda:0",
        cache_budget_bytes=engine.reservation * 10,
        workspace_reserve_bytes=1,
    )
    available = 2 * engine.reservation - materialized + 1

    def free_memory(device: torch.device) -> tuple[int, int]:
        return available, available

    monkeypatch.setattr(torch.cuda, "mem_get_info", free_memory)
    response = execute(
        engine,
        (Open(operation_id=2, session_id="b"), Open(operation_id=3, session_id="c")),
    )
    assert isinstance(response.results[0].outcome, Opened)
    assert isinstance(response.results[1].outcome, Failed)
    assert set(engine.sessions) == {"a", "b"}


def test_batch_requires_transient_cache_workspace_in_addition_to_session_reservations(
    engine: PyTorchEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    execute(engine, (Open(operation_id=0, session_id="a"), Open(operation_id=1, session_id="b")))
    appended = 3 + pseudo_token_count(1600) + 9
    workspace = batch_workspace_bytes(engine.model.text_config, (0, 0), appended)
    available = (
        engine.configuration.workspace_reserve_bytes + 2 * engine.reservation + workspace - 1
    )

    def free_memory(device: torch.device) -> tuple[int, int]:
        return available, available

    monkeypatch.setattr(torch.cuda, "mem_get_info", free_memory)
    response = execute(
        engine,
        (prefill("a", 2, 1, 1, 0, 3200), prefill("b", 3, 1, 1, 3200, 3200)),
        bytes(6400),
    )
    assert all(isinstance(result.outcome, Failed) for result in response.results)
    assert all(session.cache.get_seq_length() == 0 for session in engine.sessions.values())


def test_invalid_utf8_model_tokens_do_not_invalidate_any_session(
    engine: PyTorchEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    def invalid_piece(token_id: int) -> bytes:
        return b"\xff"

    monkeypatch.setattr(engine.model, "token_piece", invalid_piece)
    execute(engine, (Open(operation_id=0, session_id="a"), Open(operation_id=1, session_id="b")))
    response = execute(
        engine,
        (prefill("a", 2, 1, 1, 0, 3200), prefill("b", 3, 1, 1, 3200, 3200)),
        bytes(6400),
    )
    for result in response.results:
        assert isinstance(result.outcome, Token)
        assert result.outcome.text_delta == "�"
    assert set(engine.sessions) == {"a", "b"}

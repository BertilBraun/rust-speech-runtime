import json

import pytest
from pydantic import ValidationError

from voice_worker.protocol import Decode, Open, Prefill, Request


def test_rust_wire_json_roundtrip() -> None:
    encoded = json.dumps(
        {
            "request_id": 4,
            "body_bytes": 3200,
            "operations": [
                {
                    "type": "prefill",
                    "operation_id": 8,
                    "session_id": "9:session",
                    "turn_id": 2,
                    "generation": 3,
                    "audio_offset": 0,
                    "audio_bytes": 3200,
                    "accepted": {"turn_id": 1, "index": 2, "token_id": 44},
                }
            ],
        }
    )
    parsed = Request.model_validate_json(encoded)
    assert isinstance(parsed.operations[0], Prefill)
    assert Request.model_validate_json(parsed.model_dump_json()) == parsed


@pytest.mark.parametrize(
    "operations,body_bytes",
    [
        ((Open(operation_id=1, session_id="same"), Open(operation_id=2, session_id="same")), 0),
        ((Open(operation_id=1, session_id="a"), Open(operation_id=1, session_id="b")), 0),
        (
            (
                Open(operation_id=1, session_id="a"),
                Prefill(
                    operation_id=2,
                    session_id="b",
                    turn_id=1,
                    generation=1,
                    audio_offset=0,
                    audio_bytes=2,
                ),
            ),
            2,
        ),
        (
            (
                Prefill(
                    operation_id=1,
                    session_id="a",
                    turn_id=1,
                    generation=1,
                    audio_offset=1,
                    audio_bytes=2,
                ),
            ),
            3,
        ),
        (
            (
                Prefill(
                    operation_id=1,
                    session_id="a",
                    turn_id=1,
                    generation=1,
                    audio_offset=0,
                    audio_bytes=3,
                ),
            ),
            3,
        ),
        ((Open(operation_id=1, session_id="a"),), 2),
    ],
)
def test_reject_malformed_batches(operations: tuple[Open | Prefill, ...], body_bytes: int) -> None:
    with pytest.raises(ValidationError):
        Request(request_id=1, body_bytes=body_bytes, operations=operations)


def test_unknown_fields_and_missing_acceptance_rejected() -> None:
    with pytest.raises(ValidationError):
        Open.model_validate_json('{"type":"open","operation_id":1,"session_id":"a","extra":2}')
    with pytest.raises(ValidationError):
        Decode.model_validate_json(
            '{"type":"decode","operation_id":1,"session_id":"a","turn_id":1,"generation":1}'
        )

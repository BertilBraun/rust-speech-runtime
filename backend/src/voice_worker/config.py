"""Strict deployment configuration with explicit checkpoint identity."""

from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator

from voice_worker.protocol import Record


class ModelManifest(Record):
    projector_checkpoint: Path
    projector_sha256: Annotated[str, Field(pattern="^[a-f0-9]{64}$")]
    language_model: str = "Qwen/Qwen3.5-2B"
    language_revision: str = "15852e8c16360a2fea060d615a32b45270f8a8fc"
    encoder_model: str = "openai/whisper-small"
    encoder_revision: str = "973afd24965f72e36ca33b3055d56a652f456b4d"
    prompt_policy: str = "persistent-speech-chat-v1"
    decoding_policy: str = "greedy-v1"

    @model_validator(mode="after")
    def validate_policy(self) -> "ModelManifest":
        if self.prompt_policy != "persistent-speech-chat-v1" or self.decoding_policy != "greedy-v1":
            raise ValueError("This adapter implements persistent-speech-chat-v1 and greedy-v1")
        return self


class WorkerConfig(Record):
    model: ModelManifest
    device: Annotated[str, Field(pattern="^cuda(?::[0-9]+)?$")]
    host: str = "127.0.0.1"
    port: Annotated[int, Field(ge=1, le=65535)] = 9100
    max_sessions: Annotated[int, Field(gt=0)] = 64
    max_context_tokens: Annotated[int, Field(ge=512)] = 4096
    max_batch_size: Annotated[int, Field(gt=0)] = 16
    cache_budget_bytes: Annotated[int, Field(gt=0)] = 8 * 1024**3
    workspace_reserve_bytes: Annotated[int, Field(gt=0)] = 2 * 1024**3
    allocator_memory_fraction: Annotated[float, Field(gt=0, le=1)] = 1.0
    max_metadata_bytes: Annotated[int, Field(gt=0)] = 1024 * 1024
    connection_timeout_seconds: Annotated[float, Field(gt=0)] = 300.0

    @property
    def max_body_bytes(self) -> int:
        return self.max_batch_size * 480_000 * 2

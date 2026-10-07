"""Bounded real-model decode timings and an optional CPU/CUDA operator trace."""

import argparse
import math
import time
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict
from torch.profiler import ProfilerActivity, profile, record_function
from transformers.cache_utils import DynamicCache

from voice_worker.config import WorkerConfig
from voice_worker.model import NEWLINE, ForwardBatch, SpeechModel


class DecodeMeasurement(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_size: int
    initial_context_tokens: int
    steps: int
    wall_ms: tuple[float, ...]
    median_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    tokens_per_second: float


class DecodeReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    model_id: str
    measurements: tuple[DecodeMeasurement, ...]
    peak_allocated_bytes: int
    peak_reserved_bytes: int


def start_batch(model: SpeechModel, batch_size: int, context_tokens: int) -> ForwardBatch:
    prompt = model.embed((NEWLINE,) * context_tokens)
    caches = tuple(DynamicCache(config=model.text_config) for _ in range(batch_size))
    return model.forward_batch((prompt,) * batch_size, caches)


def decode_step(model: SpeechModel, batch: ForwardBatch) -> ForwardBatch:
    with record_function("decode.embed"):
        embeddings = tuple(model.embed((token,)) for token in batch.token_ids)
    with record_function("decode.forward"):
        return model.forward_batch(embeddings, batch.caches)


def measure(
    model: SpeechModel, batch_size: int, context_tokens: int, steps: int
) -> DecodeMeasurement:
    batch = start_batch(model, batch_size, context_tokens)
    for _ in range(5):
        batch = decode_step(model, batch)
    durations = []
    for _ in range(steps):
        started = time.perf_counter()
        batch = decode_step(model, batch)
        durations.append((time.perf_counter() - started) * 1000)
    ordered = sorted(durations)
    return DecodeMeasurement(
        batch_size=batch_size,
        initial_context_tokens=context_tokens,
        steps=steps,
        wall_ms=tuple(durations),
        median_ms=ordered[len(ordered) // 2],
        p95_ms=ordered[math.ceil(len(ordered) * 0.95) - 1],
        p99_ms=ordered[math.ceil(len(ordered) * 0.99) - 1],
        max_ms=ordered[-1],
        tokens_per_second=batch_size * steps * 1000 / sum(durations),
    )


def trace_decode(model: SpeechModel, output: Path) -> None:
    batch = start_batch(model, min(16, model.configuration.max_batch_size), 256)
    for _ in range(5):
        batch = decode_step(model, batch)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        for _ in range(3):
            batch = decode_step(model, batch)
    profiler.export_chrome_trace(str(output / "decode-trace.json"))
    operators = profiler.key_averages()
    for sort_key in ("self_cpu_time_total", "self_cuda_time_total"):
        (output / f"operators-{sort_key}.txt").write_text(
            operators.table(sort_by=sort_key, row_limit=40), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--trace", action="store_true")
    arguments = parser.parse_args()
    if not 1 <= arguments.steps <= 100:
        parser.error("steps must be between 1 and 100")
    configuration = WorkerConfig.model_validate_json(arguments.config.read_text(encoding="utf-8"))
    arguments.output.mkdir(parents=True, exist_ok=True)
    model = SpeechModel(configuration)
    with torch.inference_mode():
        measurements = tuple(
            measure(model, size, context, arguments.steps)
            for context in (64, 256)
            for size in (1, 8, 16)
            if size <= configuration.max_batch_size
        )
        report = DecodeReport(
            label=arguments.label,
            model_id=model.model_id,
            measurements=measurements,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(model.device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(model.device),
        )
        (arguments.output / "decode.json").write_text(report.model_dump_json(indent=2))
        print(report.model_dump_json(indent=2), flush=True)
        if arguments.trace:
            trace_decode(model, arguments.output)


if __name__ == "__main__":
    main()

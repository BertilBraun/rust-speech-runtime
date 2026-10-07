"""Copy one completed training save without modifying its source run."""

import argparse
import hashlib
import time
from pathlib import Path

from voice_worker.config import ModelManifest, WorkerConfig


def write_worker_configuration(checkpoint: Path, checksum: str) -> Path:
    configuration = WorkerConfig(
        model=ModelManifest(projector_checkpoint=checkpoint.resolve(), projector_sha256=checksum),
        device="cuda:0",
        max_sessions=1,
        max_context_tokens=2048,
        max_batch_size=1,
        cache_budget_bytes=128 * 1024**2,
        workspace_reserve_bytes=512 * 1024**2,
        allocator_memory_fraction=0.25,
    )
    configuration_path = checkpoint.parent / "worker.json"
    configuration_path.write_text(configuration.model_dump_json(indent=2), encoding="utf-8")
    return configuration_path


def copy_checkpoint(source_run: Path, destination: Path) -> Path:
    checkpoint = source_run / "checkpoint" / "projector.safetensors"
    state = source_run / "checkpoint" / "state.json"
    for _ in range(3):
        before = checkpoint.stat()
        state_before = state.read_bytes()
        if before.st_mtime_ns > state.stat().st_mtime_ns:
            time.sleep(0.2)
            continue
        weights = checkpoint.read_bytes()
        after = checkpoint.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or state_before != state.read_bytes():
            time.sleep(0.2)
            continue
        destination.mkdir(parents=True, exist_ok=False)
        copied = destination / "projector.safetensors"
        copied.write_bytes(weights)
        (destination / "training_state.json").write_bytes(state_before)
        (destination / "training_config.json").write_bytes(
            (source_run / "config.json").read_bytes()
        )
        return write_worker_configuration(copied, hashlib.sha256(weights).hexdigest())
    raise ValueError("Training checkpoint changed during capture; retry after a completed save")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    arguments = parser.parse_args()
    print(copy_checkpoint(arguments.source_run, arguments.destination))


if __name__ == "__main__":
    main()

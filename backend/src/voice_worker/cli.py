"""Production worker entry point; fixtures have a separate executable."""

import argparse
import asyncio
import logging
from pathlib import Path

from voice_worker.config import WorkerConfig
from voice_worker.framing import FrameLimits
from voice_worker.model import SpeechModel
from voice_worker.pytorch_engine import PyTorchEngine
from voice_worker.server import WorkerServer


async def serve(configuration: WorkerConfig, engine: PyTorchEngine) -> None:
    worker = WorkerServer(
        engine,
        FrameLimits(
            configuration.max_metadata_bytes,
            configuration.max_body_bytes,
            configuration.connection_timeout_seconds,
        ),
    )
    server = await asyncio.start_server(worker.handle, configuration.host, configuration.port)
    try:
        async with server:
            await server.serve_forever()
    finally:
        await worker.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent speech-model GPU worker")
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    configuration = WorkerConfig.model_validate_json(arguments.config.read_text(encoding="utf-8"))
    logging.basicConfig(level=logging.INFO)
    model = SpeechModel(configuration)
    model.warmup()
    asyncio.run(serve(configuration, PyTorchEngine(configuration, model)))


if __name__ == "__main__":
    main()

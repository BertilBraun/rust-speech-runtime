"""One gateway connection owns the worker; no concurrent mutation of model state."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import cast

from voice_worker.engine import Engine
from voice_worker.framing import FrameLimits, read_request, write_message

LOGGER = logging.getLogger(__name__)


class WorkerServer:
    def __init__(self, engine: Engine, limits: FrameLimits) -> None:
        self.engine = engine
        self.limits = limits
        self.connected = False
        self.active_handler: asyncio.Task[None] | None = None
        self.active_writer: asyncio.StreamWriter | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-owner")

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self.connected:
            writer.close()
            await writer.wait_closed()
            return
        self.connected = True
        self.active_handler = cast(asyncio.Task[None], asyncio.current_task())
        self.active_writer = writer
        loop = asyncio.get_running_loop()
        try:
            await write_message(writer, self.engine.ready(), self.limits)
            while True:
                request, body = await read_request(reader, self.limits)
                response = await loop.run_in_executor(
                    self.executor, self.engine.execute, request, body
                )
                await write_message(writer, response, self.limits)
        except asyncio.IncompleteReadError:
            pass
        except (ValueError, TimeoutError, ConnectionError):
            LOGGER.exception("Worker connection failed")
        finally:
            await loop.run_in_executor(self.executor, self.engine.reset)
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            self.active_handler = None
            self.active_writer = None
            self.connected = False

    async def close(self) -> None:
        if self.active_writer is not None:
            self.active_writer.close()
        try:
            if self.active_handler is not None:
                await self.active_handler
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)

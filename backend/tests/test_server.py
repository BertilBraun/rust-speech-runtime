import asyncio
import struct

import pytest
from fixture_worker import FixtureConfig, FixtureEngine

from voice_worker.framing import FrameLimits
from voice_worker.protocol import Open, Opened, Ready, Request, Response
from voice_worker.server import WorkerServer


async def read_metadata(reader: asyncio.StreamReader) -> bytes:
    length = struct.unpack(">I", await reader.readexactly(4))[0]
    return await reader.readexactly(length)


async def send_request(writer: asyncio.StreamWriter, request: Request, body: bytes = b"") -> None:
    metadata = request.model_dump_json().encode("utf-8")
    writer.write(struct.pack(">I", len(metadata)) + metadata + body)
    await writer.drain()


async def test_persistent_worker_wire_and_connection_owned_caches() -> None:
    engine = FixtureEngine(FixtureConfig())
    worker = WorkerServer(engine, FrameLimits(4096, 4096, 1.0))
    listener = await asyncio.start_server(worker.handle, "127.0.0.1", 0)
    port = listener.sockets[0].getsockname()[1]
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        ready = Ready.model_validate_json(await read_metadata(reader))
        assert ready.protocol_version == 1
        request = Request(
            request_id=1, body_bytes=0, operations=(Open(operation_id=2, session_id="a"),)
        )
        await send_request(writer, request)
        response = Response.model_validate_json(await read_metadata(reader))
        assert isinstance(response.results[0].outcome, Opened)
        assert "a" in engine.sessions
        other_reader, other_writer = await asyncio.open_connection("127.0.0.1", port)
        assert await asyncio.wait_for(other_reader.read(1), 1.0) == b""
        other_writer.close()
        await other_writer.wait_closed()
        writer.close()
        await writer.wait_closed()
        for _ in range(100):
            if not worker.connected:
                break
            await asyncio.sleep(0.01)
        assert not engine.sessions
        assert not worker.connected
    finally:
        listener.close()
        await listener.wait_closed()
        await worker.close()


@pytest.mark.parametrize("length", [0, 4097, 2**32 - 1])
async def test_oversize_metadata_closes_connection_before_allocating(length: int) -> None:
    worker = WorkerServer(FixtureEngine(FixtureConfig()), FrameLimits(4096, 4096, 1.0))
    listener = await asyncio.start_server(worker.handle, "127.0.0.1", 0)
    try:
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", listener.sockets[0].getsockname()[1]
        )
        await read_metadata(reader)
        writer.write(struct.pack(">I", length))
        await writer.drain()
        assert await asyncio.wait_for(reader.read(1), 1.0) == b""
        writer.close()
        await writer.wait_closed()
    finally:
        listener.close()
        await listener.wait_closed()
        await worker.close()

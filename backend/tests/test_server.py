import asyncio
import struct
from contextlib import suppress

import pytest
from fixture_worker import FixtureConfig, FixtureEngine

from voice_worker.framing import FrameLimits
from voice_worker.protocol import Close, Closed, Open, Opened, Prefill, Ready, Request, Response
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


async def test_idle_connection_preserves_cache_beyond_frame_timeout() -> None:
    engine = FixtureEngine(FixtureConfig())
    worker = WorkerServer(engine, FrameLimits(4096, 4096, 0.05))
    listener = await asyncio.start_server(worker.handle, "127.0.0.1", 0)
    try:
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", listener.sockets[0].getsockname()[1]
        )
        await read_metadata(reader)
        await asyncio.sleep(0.2)
        opened = Request(
            request_id=1, body_bytes=0, operations=(Open(operation_id=1, session_id="a"),)
        )
        await send_request(writer, opened)
        response = Response.model_validate_json(await read_metadata(reader))
        assert isinstance(response.results[0].outcome, Opened)
        await asyncio.sleep(0.2)
        assert worker.connected
        assert "a" in engine.sessions
        closed = Request(
            request_id=2, body_bytes=0, operations=(Close(operation_id=2, session_id="a"),)
        )
        await send_request(writer, closed)
        response = Response.model_validate_json(await read_metadata(reader))
        assert isinstance(response.results[0].outcome, Closed)
        assert not engine.sessions
        writer.close()
        await writer.wait_closed()
    finally:
        listener.close()
        await listener.wait_closed()
        await worker.close()


@pytest.mark.parametrize("incomplete", ["header", "metadata", "body"])
async def test_partial_frame_still_times_out(incomplete: str) -> None:
    worker = WorkerServer(FixtureEngine(FixtureConfig()), FrameLimits(4096, 4096, 0.05))
    listener = await asyncio.start_server(worker.handle, "127.0.0.1", 0)
    try:
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", listener.sockets[0].getsockname()[1]
        )
        await read_metadata(reader)
        request = Request(
            request_id=1,
            body_bytes=3200,
            operations=(
                Prefill(
                    operation_id=2,
                    session_id="a",
                    turn_id=1,
                    generation=1,
                    audio_offset=0,
                    audio_bytes=3200,
                ),
            ),
        )
        metadata = request.model_dump_json().encode("utf-8")
        match incomplete:
            case "header":
                fragment = b"\x00"
            case "metadata":
                fragment = struct.pack(">I", len(metadata)) + metadata[:1]
            case "body":
                fragment = struct.pack(">I", len(metadata)) + metadata + b"\x00"
        writer.write(fragment)
        await writer.drain()
        assert await asyncio.wait_for(reader.read(1), 1.0) == b""
        writer.close()
        await writer.wait_closed()
    finally:
        listener.close()
        await listener.wait_closed()
        await worker.close()


async def test_worker_close_releases_idle_connection_and_caches() -> None:
    engine = FixtureEngine(FixtureConfig())
    worker = WorkerServer(engine, FrameLimits(4096, 4096, 0.05))
    listener = await asyncio.start_server(worker.handle, "127.0.0.1", 0)
    try:
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", listener.sockets[0].getsockname()[1]
        )
        await read_metadata(reader)
        await send_request(
            writer,
            Request(request_id=1, body_bytes=0, operations=(Open(operation_id=1, session_id="a"),)),
        )
        await read_metadata(reader)
        await asyncio.wait_for(worker.close(), 1.0)
        assert not worker.connected
        assert not engine.sessions
        assert await reader.read(1) == b""
        writer.close()
        await writer.wait_closed()
    finally:
        listener.close()
        await listener.wait_closed()
        await worker.close()


async def test_partial_frame_uses_one_budget_across_header_and_body() -> None:
    worker = WorkerServer(FixtureEngine(FixtureConfig()), FrameLimits(4096, 4096, 0.2))
    listener = await asyncio.start_server(worker.handle, "127.0.0.1", 0)
    try:
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", listener.sockets[0].getsockname()[1]
        )
        await read_metadata(reader)
        request = Request(
            request_id=1,
            body_bytes=3200,
            operations=(
                Prefill(
                    operation_id=1,
                    session_id="a",
                    turn_id=1,
                    generation=1,
                    audio_offset=0,
                    audio_bytes=3200,
                ),
            ),
        )
        metadata = request.model_dump_json().encode("utf-8")
        header = struct.pack(">I", len(metadata))
        writer.write(header[:1])
        await writer.drain()
        await asyncio.sleep(0.12)
        writer.write(header[1:] + metadata + bytes(1600))
        await writer.drain()
        await asyncio.sleep(0.12)
        with suppress(ConnectionError):
            writer.write(bytes(1600))
            await writer.drain()
        assert await asyncio.wait_for(reader.read(1), 1.0) == b""
        writer.close()
        with suppress(ConnectionError):
            await writer.wait_closed()
    finally:
        listener.close()
        await listener.wait_closed()
        await worker.close()

"""Bounded length-prefixed JSON followed by an exact binary payload."""

import asyncio
import struct
from dataclasses import dataclass

from voice_worker.protocol import Ready, Request, Response


@dataclass(frozen=True)
class FrameLimits:
    metadata_bytes: int
    body_bytes: int
    timeout_seconds: float


async def read_request(reader: asyncio.StreamReader, limits: FrameLimits) -> tuple[Request, bytes]:
    async def read_frame() -> tuple[Request, bytes]:
        length = struct.unpack(">I", await reader.readexactly(4))[0]
        if length == 0 or length > limits.metadata_bytes:
            raise ValueError("Metadata frame exceeds its configured bound")
        request = Request.model_validate_json(await reader.readexactly(length))
        if request.body_bytes > limits.body_bytes:
            raise ValueError("Audio body exceeds its configured bound")
        body = await reader.readexactly(request.body_bytes)
        return request, body

    return await asyncio.wait_for(read_frame(), timeout=limits.timeout_seconds)


async def write_message(
    writer: asyncio.StreamWriter, message: Ready | Response, limits: FrameLimits
) -> None:
    metadata = message.model_dump_json().encode("utf-8")
    if len(metadata) > limits.metadata_bytes:
        raise ValueError("Response metadata exceeds its configured bound")
    writer.write(struct.pack(">I", len(metadata)))
    writer.write(metadata)
    await asyncio.wait_for(writer.drain(), timeout=limits.timeout_seconds)

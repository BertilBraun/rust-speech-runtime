import asyncio
import sys
from pathlib import Path

import pytest
from test_server import read_metadata

from voice_worker.protocol import Ready


@pytest.mark.parametrize("exit_on_disconnect", [False, True])
async def test_fixture_cli_disconnect_lifecycle(exit_on_disconnect: bool) -> None:
    arguments = ("--exit-on-disconnect",) if exit_on_disconnect else ()
    fixture = Path(__file__).with_name("fixture_worker.py")
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(fixture),
        "--port",
        "0",
        *arguments,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdout is not None
    try:
        ready_line = await asyncio.wait_for(process.stdout.readline(), 10.0)
        port = int(ready_line.decode("utf-8").strip().rsplit(":", 1)[1])
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        ready = Ready.model_validate_json(await read_metadata(reader))
        assert ready.model_id == "benchmark-fixture-v1"
        rejected_reader, rejected_writer = await asyncio.open_connection("127.0.0.1", port)
        assert await asyncio.wait_for(rejected_reader.read(1), 1.0) == b""
        rejected_writer.close()
        await rejected_writer.wait_closed()
        assert process.returncode is None
        writer.close()
        await writer.wait_closed()
        if exit_on_disconnect:
            assert await asyncio.wait_for(process.wait(), 5.0) == 0
            _, errors = await process.communicate()
            assert errors == b""
        else:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(process.wait(), 0.1)
            new_reader, new_writer = await asyncio.open_connection("127.0.0.1", port)
            Ready.model_validate_json(await read_metadata(new_reader))
            new_writer.close()
            await new_writer.wait_closed()
            assert process.returncode is None
    finally:
        if process.returncode is None:
            process.terminate()
            await process.wait()

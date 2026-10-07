import subprocess
from typing import cast
from unittest.mock import Mock

import guard_worker
import pytest


@pytest.mark.parametrize("failure", [FileNotFoundError, ValueError])
@pytest.mark.parametrize("exit_status", [None, 0, 7])
def test_resource_read_race_preserves_child_exit_status(
    monkeypatch: pytest.MonkeyPatch,
    failure: type[FileNotFoundError] | type[ValueError],
    exit_status: int | None,
) -> None:
    child = Mock(spec=subprocess.Popen)
    child.pid = 123
    child.poll.return_value = exit_status

    def failed_observation(child_pid: int) -> guard_worker.Resources:
        assert child_pid == 123
        raise failure("Process exited during resource sampling")

    monkeypatch.setattr(guard_worker, "resources", failed_observation)
    process = cast(subprocess.Popen[bytes], child)
    if exit_status is None:
        with pytest.raises(failure):
            guard_worker.observe_child(process)
    else:
        assert guard_worker.observe_child(process) is None

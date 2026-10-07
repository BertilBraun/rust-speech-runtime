"""Stop only the owned serving process if a shared Linux node loses headroom."""

import argparse
import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import FrameType


@dataclass(frozen=True)
class GuardLimits:
    minimum_free_vram_mib: int
    minimum_available_ram_mib: int
    maximum_child_rss_mib: int

    def __post_init__(self) -> None:
        if (
            min(
                self.minimum_free_vram_mib,
                self.minimum_available_ram_mib,
                self.maximum_child_rss_mib,
            )
            <= 0
        ):
            raise ValueError("Resource limits must be positive")


@dataclass(frozen=True)
class Resources:
    timestamp: float
    free_vram_mib: int
    available_ram_mib: int
    child_rss_mib: int
    cgroup_usage_mib: int

    def violation(self, limits: GuardLimits) -> bool:
        return (
            self.free_vram_mib < limits.minimum_free_vram_mib
            or self.available_ram_mib < limits.minimum_available_ram_mib
            or self.child_rss_mib > limits.maximum_child_rss_mib
        )


def statistic(path: Path, name: str) -> int:
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields[0].rstrip(":") == name:
            return int(fields[1])
    raise ValueError(f"Missing {name} in {path}")


def resources(child_pid: int) -> Resources:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    memory = Path("/sys/fs/cgroup/memory")
    limit = int((memory / "memory.limit_in_bytes").read_text(encoding="utf-8"))
    usage = int((memory / "memory.usage_in_bytes").read_text(encoding="utf-8"))
    reclaimable = statistic(memory / "memory.stat", "total_inactive_file")
    available = min(
        statistic(Path("/proc/meminfo"), "MemAvailable") * 1024,
        max(0, limit - usage + reclaimable),
    )
    return Resources(
        timestamp=time.time(),
        free_vram_mib=int(gpu.stdout.strip()),
        available_ram_mib=available // 1024**2,
        child_rss_mib=statistic(Path(f"/proc/{child_pid}/status"), "VmRSS") // 1024,
        cgroup_usage_mib=usage // 1024**2,
    )


def run(command: list[str], limits: GuardLimits, report: Path) -> int:
    stopped = threading.Event()

    def request_stop(signum: int, frame: FrameType | None) -> None:
        stopped.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    os.nice(19)
    with report.open("a", encoding="utf-8") as output:
        child = subprocess.Popen(command)
        try:
            while child.poll() is None and not stopped.is_set():
                try:
                    observation = resources(child.pid)
                except FileNotFoundError:
                    break
                output.write(json.dumps(asdict(observation)) + "\n")
                output.flush()
                if observation.violation(limits):
                    print(f"Serving stopped by resource guard: {observation}", flush=True)
                    return 1
                stopped.wait(2)
            return 0 if stopped.is_set() else child.wait(timeout=15)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--minimum-free-vram-mib", type=int, required=True)
    parser.add_argument("--minimum-available-ram-mib", type=int, required=True)
    parser.add_argument("--maximum-child-rss-mib", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("An owned serving command is required")
    limits = GuardLimits(
        arguments.minimum_free_vram_mib,
        arguments.minimum_available_ram_mib,
        arguments.maximum_child_rss_mib,
    )
    raise SystemExit(run(command, limits, arguments.report))


if __name__ == "__main__":
    main()

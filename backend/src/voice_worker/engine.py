"""Synchronous model ownership boundary, executed on one dedicated thread."""

from abc import ABC, abstractmethod

from voice_worker.protocol import Ready, Request, Response


class Engine(ABC):
    @abstractmethod
    def ready(self) -> Ready:
        """Return capabilities only after model warmup has completed."""

    @abstractmethod
    def execute(self, request: Request, body: bytes) -> Response:
        """Execute the Rust-selected batch without advancing unaccepted tokens."""

    @abstractmethod
    def reset(self) -> None:
        """Release session caches after the owning gateway connection ends."""

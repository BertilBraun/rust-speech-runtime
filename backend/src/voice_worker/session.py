"""Accepted-token bookkeeping never mutates cache for a rejected proposal."""

from dataclasses import dataclass

from transformers.cache_utils import DynamicCache

from voice_worker.detokenizer import TextPreview
from voice_worker.model import EOS, NEWLINE
from voice_worker.protocol import AcceptedToken


@dataclass(frozen=True)
class Proposal:
    token: AcceptedToken
    preview: TextPreview


@dataclass
class ModelSession:
    cache: DynamicCache
    turn_id: int | None = None
    generation: int | None = None
    consumed: AcceptedToken | None = None
    proposed: Proposal | None = None
    pending_text: bytes = b""

    def validate_acceptance(self, accepted: AcceptedToken) -> None:
        if self.proposed is None or accepted != self.proposed.token:
            raise ValueError("Accepted token does not match the last proposal")

    def accept(self, accepted: AcceptedToken) -> None:
        self.validate_acceptance(accepted)
        assert self.proposed is not None
        self.pending_text = self.proposed.preview.pending
        self.consumed = accepted
        self.proposed = None

    def next_turn_prefix(self, accepted: AcceptedToken | None) -> tuple[int, ...]:
        if self.turn_id is None:
            if accepted is not None:
                raise ValueError("A fresh session has no accepted token to reconcile")
            return ()
        pending = ()
        final = self.consumed
        if accepted is not None:
            if accepted.turn_id != self.turn_id:
                raise ValueError("Final accepted token belongs to a different turn")
            if accepted != self.consumed:
                self.validate_acceptance(accepted)
                pending = (accepted.token_id,)
                final = accepted
        ending = () if final is not None and final.token_id == EOS else (EOS,)
        return pending + ending + (NEWLINE,)

    def start_turn(self, turn_id: int, generation: int) -> None:
        self.turn_id = turn_id
        self.generation = generation
        self.consumed = None
        self.proposed = None
        self.pending_text = b""

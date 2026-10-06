import pytest
from transformers.cache_utils import DynamicCache

from voice_worker.detokenizer import TextPreview
from voice_worker.model import EOS, NEWLINE
from voice_worker.protocol import AcceptedToken
from voice_worker.session import ModelSession, Proposal


def active_session() -> ModelSession:
    session = ModelSession(DynamicCache(), turn_id=100, generation=1)
    session.proposed = Proposal(
        AcceptedToken(turn_id=100, index=0, token_id=40), TextPreview("x", b"")
    )
    return session


def test_final_accepted_proposal_is_flushed_once() -> None:
    session = active_session()
    accepted = session.proposed.token
    assert session.next_turn_prefix(accepted) == (40, EOS, NEWLINE)
    session.accept(accepted)
    assert session.next_turn_prefix(accepted) == (EOS, NEWLINE)
    assert session.next_turn_prefix(None) == (EOS, NEWLINE)


def test_unaccepted_proposal_does_not_enter_prefix() -> None:
    session = active_session()
    assert session.next_turn_prefix(None) == (EOS, NEWLINE)
    assert session.consumed is None


def test_identical_tokens_with_different_indices_remain_distinct() -> None:
    session = active_session()
    first = session.proposed.token
    session.accept(first)
    second = AcceptedToken(turn_id=100, index=1, token_id=40)
    session.proposed = Proposal(second, TextPreview("x", b""))
    assert session.next_turn_prefix(second) == (40, EOS, NEWLINE)
    with pytest.raises(ValueError):
        session.validate_acceptance(first)


def test_eos_is_not_duplicated_on_next_user_turn() -> None:
    session = active_session()
    final = AcceptedToken(turn_id=100, index=0, token_id=EOS)
    session.proposed = Proposal(final, TextPreview("", b""))
    assert session.next_turn_prefix(final) == (EOS, NEWLINE)
    session.accept(final)
    assert session.next_turn_prefix(None) == (NEWLINE,)


def test_acceptance_and_detokenizer_commit_only_the_matching_proposal() -> None:
    session = active_session()
    accepted = session.proposed.token
    session.proposed = Proposal(accepted, TextPreview("", b"\xe2"))
    assert session.pending_text == b""
    session.accept(accepted)
    assert session.pending_text == b"\xe2"
    assert session.proposed is None
    with pytest.raises(ValueError):
        session.next_turn_prefix(AcceptedToken(turn_id=99, index=0, token_id=40))

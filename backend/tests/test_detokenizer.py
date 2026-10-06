import pytest

from voice_worker.detokenizer import BYTE_DECODER, TextPreview, preview_text, token_bytes


def test_byte_alphabet_roundtrip_all_values() -> None:
    inverse = {number: character for character, number in BYTE_DECODER.items()}
    assert token_bytes("".join(inverse[number] for number in range(256))) == bytes(range(256))


def test_split_utf8_is_buffered_until_a_complete_character() -> None:
    first = preview_text(b"", b"hello \xe2")
    assert first.delta == "hello "
    assert first.pending == b"\xe2"
    second = preview_text(first.pending, b"\x82")
    assert second.delta == ""
    third = preview_text(second.pending, b"\xac!")
    assert third.delta == "€!"
    assert third.pending == b""


def test_preview_does_not_commit_and_eos_flushes_incomplete_bytes() -> None:
    pending = b"\xe2"
    assert preview_text(pending, b"\x82\xac").delta == "€"
    assert pending == b"\xe2"
    assert preview_text(pending, b"", finished=True).delta == "�"


@pytest.mark.parametrize(
    ("pending", "piece", "delta", "remaining"),
    [
        (b"", b"\xff", "�", b""),
        (b"\xe2", b"hello", "�hello", b""),
        (b"", b"\xff\xe2", "�", b"\xe2"),
        (b"\xe2", b"\x82\xac\xff", "€�", b""),
    ],
)
def test_arbitrary_generated_bytes_replace_invalid_sequences_and_keep_only_incomplete_tail(
    pending: bytes, piece: bytes, delta: str, remaining: bytes
) -> None:
    assert preview_text(pending, piece) == TextPreview(delta, remaining)

"""Preview byte-level tokens without committing proposals to session history."""

from dataclasses import dataclass


def byte_decoder() -> dict[str, int]:
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    encoded = visible.copy()
    extra = 0
    for number in range(256):
        if number not in visible:
            visible.append(number)
            encoded.append(256 + extra)
            extra += 1
    return {chr(character): number for number, character in zip(visible, encoded, strict=True)}


BYTE_DECODER = byte_decoder()


def token_bytes(piece: str) -> bytes:
    try:
        return bytes(BYTE_DECODER[character] for character in piece)
    except KeyError as error:
        raise ValueError("Tokenizer is incompatible with the Qwen byte-level adapter") from error


@dataclass(frozen=True)
class TextPreview:
    delta: str
    pending: bytes


def preview_text(pending: bytes, piece: bytes, finished: bool = False) -> TextPreview:
    combined = pending + piece
    if finished:
        return TextPreview(combined.decode("utf-8", errors="replace"), b"")
    try:
        return TextPreview(combined.decode("utf-8"), b"")
    except UnicodeDecodeError as error:
        if error.reason != "unexpected end of data":
            raise ValueError("Tokenizer produced invalid UTF-8 bytes") from error
        return TextPreview(combined[: error.start].decode("utf-8"), combined[error.start :])

"""Preview byte-level tokens without committing proposals to session history."""

from dataclasses import dataclass
from encodings.utf_8 import IncrementalDecoder


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
    decoder = IncrementalDecoder(errors="replace")
    delta = decoder.decode(pending + piece, final=finished)
    incomplete, _ = decoder.getstate()
    return TextPreview(delta, incomplete)

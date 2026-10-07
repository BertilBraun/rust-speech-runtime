import numpy as np
import pytest
import torch

from voice_worker.model import pcm_waveform, pseudo_token_count
from voice_worker.projector import SpeechProjector, mean_pool


@pytest.mark.parametrize(
    "samples,expected", [(1, 1), (320, 1), (1600, 1), (1601, 2), (480000, 300)]
)
def test_pseudo_tokens_follow_real_encoder_count(samples: int, expected: int) -> None:
    assert pseudo_token_count(samples) == expected


@pytest.mark.parametrize("packet", [b"", b"\x00", bytes(960002)], ids=["empty", "odd", "oversize"])
def test_audio_boundaries(packet: bytes) -> None:
    with pytest.raises(ValueError):
        pcm_waveform(packet)


def test_pcm_signed_scale_not_packet_normalized() -> None:
    raw = np.array([-32768, -16384, 0, 8192, 32767], dtype="<i2")
    np.testing.assert_array_equal(pcm_waveform(raw.tobytes()), raw.astype(np.float32) / 32768)


def test_last_pool_block_uses_real_frames() -> None:
    features = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    expected = torch.stack((features[:5].mean(0), features[5:].mean(0)))
    torch.testing.assert_close(mean_pool(features), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_exact_projector_checkpoint_layout_and_dtype(dtype: torch.dtype) -> None:
    projector = SpeechProjector().to(dtype=dtype)
    assert sum(parameter.numel() for parameter in projector.parameters()) == 2888192
    assert set(projector.state_dict()) == {
        "normalization.weight",
        "normalization.bias",
        "projection.0.weight",
        "projection.0.bias",
        "projection.2.weight",
        "projection.2.bias",
    }
    assert projector(torch.ones(7, 768, dtype=torch.bfloat16)).dtype == dtype
    assert all(parameter.dtype == dtype for parameter in projector.parameters())

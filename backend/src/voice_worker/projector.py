"""The trained FP32 LayerNorm, pool-five, MLP speech interface."""

import torch
from torch import Tensor, nn
from torch.nn import functional


def mean_pool(features: Tensor, factor: int = 5) -> Tensor:
    if features.shape[-2] == 0 or factor <= 0:
        raise ValueError("Pooling requires nonempty features and a positive factor")
    padding = (-features.shape[-2]) % factor
    padded = functional.pad(features, (0, 0, 0, padding))
    pooled = padded.reshape(*features.shape[:-2], -1, factor, features.shape[-1]).sum(-2)
    counts = torch.full((pooled.shape[-2],), factor, device=features.device, dtype=features.dtype)
    counts[-1] = factor - padding
    return pooled / counts.unsqueeze(-1)


class SpeechProjector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(768, eps=1e-5)
        self.projection = nn.Sequential(nn.Linear(768, 1024), nn.GELU(), nn.Linear(1024, 2048))

    def forward(self, features: Tensor) -> Tensor:
        normalized = self.normalization(features.to(torch.float32))
        return self.projection(mean_pool(normalized))

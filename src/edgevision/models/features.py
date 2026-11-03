"""Input block shared by every model, so ONNX export carries velocity and standardisation."""

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from edgevision.layout import COORDS_PER_LANDMARK, FRAME_DIM, NUM_COORDS, NUM_FRAMES, NUM_LANDMARKS

FEATURE_DIM = FRAME_DIM + NUM_COORDS
STD_FLOOR = 1e-6
STATS_CHUNK = 4096


class FeatureBlock(nn.Module):
    """(B, 32, 100) -> (B, 32, 198): the input frames plus coordinate velocity, standardised.

    Velocity is the frame difference of the 98 coordinates, zero at t = 0 and zero wherever
    either endpoint point is missing (x == y == 0.0).
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(FEATURE_DIM))
        self.register_buffer("std", torch.ones(FEATURE_DIM))

    def raw_features(self, x: torch.Tensor) -> torch.Tensor:
        points = x[..., :NUM_COORDS].reshape(-1, NUM_FRAMES, NUM_LANDMARKS, COORDS_PER_LANDMARK)
        present = (points[..., 0] != 0.0) | (points[..., 1] != 0.0)
        both_present = present[:, 1:] & present[:, :-1]
        step = (points[:, 1:] - points[:, :-1]) * both_present.unsqueeze(-1).to(points.dtype)
        velocity = torch.cat([torch.zeros_like(step[:, :1]), step], dim=1)
        return torch.cat([x, velocity.reshape(-1, NUM_FRAMES, NUM_COORDS)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.raw_features(x) - self.mean) / self.std.clamp_min(STD_FLOOR)


class SignClassifier(nn.Module):
    """Base for every model: (B, 32, 100) landmarks in, (B, num_classes) logits out."""

    def __init__(self) -> None:
        super().__init__()
        self.features = FeatureBlock()


@torch.no_grad()
def compute_feature_stats(
    feature_block: FeatureBlock, x_train: torch.Tensor | npt.NDArray[np.float32]
) -> None:
    """Set the block's mean and std from every frame of ``x_train``, the train split only.

    Sums accumulate in float64 over chunks, so the full (N, 32, 198) array never exists.
    """
    data = torch.as_tensor(x_train)
    if data.shape[0] == 0:
        raise ValueError("cannot compute feature statistics from an empty train split")
    device = feature_block.mean.device
    total = torch.zeros(FEATURE_DIM, dtype=torch.float64, device=device)
    squares = torch.zeros(FEATURE_DIM, dtype=torch.float64, device=device)
    for start in range(0, data.shape[0], STATS_CHUNK):
        chunk = data[start : start + STATS_CHUNK].to(device=device, dtype=torch.float32)
        values = feature_block.raw_features(chunk).reshape(-1, FEATURE_DIM).double()
        total += values.sum(dim=0)
        squares += values.square().sum(dim=0)
    count = data.shape[0] * NUM_FRAMES
    mean = total / count
    std = (squares / count - mean.square()).clamp_min(0.0).sqrt().clamp_min(STD_FLOOR)
    feature_block.mean.copy_(mean)
    feature_block.std.copy_(std)

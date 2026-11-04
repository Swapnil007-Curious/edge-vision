"""Input blocks shared by every model, so ONNX export carries feature extraction and
standardisation. FeatureBlock is the v1 feature set; features_v2.FeatureBlockV2 is v2."""

from typing import ClassVar, Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from edgevision.layout import FRAME_DIM, NUM_COORDS, V1_LAYOUT, InputLayout

FeatureSet = Literal["v1", "v2"]
FEATURE_SETS: tuple[FeatureSet, ...] = ("v1", "v2")
FEATURE_DIM = FRAME_DIM + NUM_COORDS  # v1 features on the v1 layout: 198.
STD_FLOOR = 1e-6
STATS_CHUNK = 4096


class StandardisedFeatures(nn.Module):
    """Base for feature blocks: raw per-frame features, standardised by the mean and std
    buffers that compute_feature_stats sets from the train split."""

    feature_set: ClassVar[FeatureSet]
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, layout: InputLayout, output_dim: int) -> None:
        super().__init__()
        self.layout = layout
        self.output_dim = output_dim
        self.register_buffer("mean", torch.zeros(output_dim))
        self.register_buffer("std", torch.ones(output_dim))

    def raw_features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, frame_dim) -> (B, T, output_dim), before standardisation."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.raw_features(x) - self.mean) / self.std.clamp_min(STD_FLOOR)


class FeatureBlock(StandardisedFeatures):
    """(B, 32, 100) -> (B, 32, 198) on the v1 layout: the input frames plus coordinate velocity,
    standardised.

    Velocity is the frame difference of the coordinates, zero at t = 0 and zero wherever either
    endpoint point is missing (x == y == 0.0).
    """

    feature_set = "v1"

    def __init__(self, layout: InputLayout = V1_LAYOUT) -> None:
        super().__init__(layout, layout.frame_dim + layout.num_coords)

    def raw_features(self, x: torch.Tensor) -> torch.Tensor:
        layout = self.layout
        frames, landmarks, coords = layout.num_frames, layout.num_landmarks, layout.num_coords
        points = x[..., :coords].reshape(-1, frames, landmarks, 2)
        present = (points[..., 0] != 0.0) | (points[..., 1] != 0.0)
        both_present = present[:, 1:] & present[:, :-1]
        step = (points[:, 1:] - points[:, :-1]) * both_present.unsqueeze(-1).to(points.dtype)
        velocity = torch.cat([torch.zeros_like(step[:, :1]), step], dim=1)
        return torch.cat([x, velocity.reshape(-1, frames, coords)], dim=-1)


class SignClassifier(nn.Module):
    """Base for every model: (B, 32, frame_dim) landmarks in, (B, num_classes) logits out."""

    def __init__(self, features: StandardisedFeatures | None = None) -> None:
        super().__init__()
        self.features = FeatureBlock() if features is None else features

    def set_training_progress(self, epoch: int, epochs: int) -> None:
        """Called before each training epoch (1-based) of ``epochs``; models that schedule
        something by epoch, such as the hybrid's late dropout, override it."""


@torch.no_grad()
def compute_feature_stats(
    feature_block: StandardisedFeatures, x_train: torch.Tensor | npt.NDArray[np.float32]
) -> None:
    """Set the block's mean and std from every frame of ``x_train``, the train split only.

    Sums accumulate in float64 over chunks, so the full (N, T, output_dim) array never exists.
    """
    data = torch.as_tensor(x_train)
    if data.shape[0] == 0:
        raise ValueError("cannot compute feature statistics from an empty train split")
    device = feature_block.mean.device
    dim = feature_block.output_dim
    total = torch.zeros(dim, dtype=torch.float64, device=device)
    squares = torch.zeros(dim, dtype=torch.float64, device=device)
    for start in range(0, data.shape[0], STATS_CHUNK):
        chunk = data[start : start + STATS_CHUNK].to(device=device, dtype=torch.float32)
        values = feature_block.raw_features(chunk).reshape(-1, dim).double()
        total += values.sum(dim=0)
        squares += values.square().sum(dim=0)
    count = data.shape[0] * data.shape[1]
    mean = total / count
    std = (squares / count - mean.square()).clamp_min(0.0).sqrt().clamp_min(STD_FLOOR)
    feature_block.mean.copy_(mean)
    feature_block.std.copy_(std)

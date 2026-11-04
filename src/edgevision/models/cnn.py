"""Temporal CNN: depthwise-separable residual blocks over time, then global average pooling."""

import torch
from torch import nn

from edgevision.models.config import CNNConfig
from edgevision.models.features import SignClassifier, StandardisedFeatures


class ResidualBlock(nn.Module):
    """Depthwise conv -> BN -> GELU -> pointwise conv -> BN -> dropout, residual add, GELU."""

    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.activation(x + self.body(x))
        return out


class TemporalCNN(SignClassifier):
    def __init__(
        self, num_classes: int, cfg: CNNConfig, features: StandardisedFeatures | None = None
    ) -> None:
        super().__init__(features)
        self.proj = nn.Linear(self.features.output_dim, cfg.channels)
        self.blocks = nn.Sequential(
            *(
                ResidualBlock(cfg.channels, cfg.kernel_size, cfg.block_dropout)
                for _ in range(cfg.num_blocks)
            )
        )
        self.head = nn.Sequential(
            nn.Dropout(cfg.head_dropout), nn.Linear(cfg.channels, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels_first = self.proj(self.features(x)).transpose(1, 2)
        pooled = self.blocks(channels_first).mean(dim=-1)
        logits: torch.Tensor = self.head(pooled)
        return logits

"""CNN-transformer hybrid: a projection, ``num_stages`` x (conv blocks + one transformer block),
global average pooling over time, late dropout and a linear head.

The transformer block is transformer.EncoderLayer, the same pre-norm block with hand-written
attention that TinyTransformer uses (no nn.MultiheadAttention).
"""

import torch
from torch import nn
from torch.nn import functional

from edgevision.models.config import HybridConfig, TransformerConfig
from edgevision.models.features import SignClassifier, StandardisedFeatures
from edgevision.models.transformer import EncoderLayer


class ConvBlock(nn.Module):
    """Depthwise conv (same padding) -> BN -> SiLU -> pointwise expand -> SiLU -> pointwise
    project -> dropout, added to the input. Works on (B, C, T)."""

    def __init__(self, cfg: HybridConfig) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            cfg.dim, cfg.dim, cfg.kernel_size, padding=cfg.kernel_size // 2, groups=cfg.dim
        )
        self.norm = nn.BatchNorm1d(cfg.dim)
        self.expand = nn.Conv1d(cfg.dim, cfg.expansion_dim, 1)
        self.project = nn.Conv1d(cfg.expansion_dim, cfg.dim, 1)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(cfg.conv_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.norm(self.depthwise(x)))
        hidden = self.dropout(self.project(self.activation(self.expand(hidden))))
        out: torch.Tensor = x + hidden
        return out


class Stage(nn.Module):
    """``conv_blocks`` conv blocks over time, then one transformer block. Works on (B, T, C)."""

    def __init__(self, cfg: HybridConfig) -> None:
        super().__init__()
        self.convs = nn.Sequential(*(ConvBlock(cfg) for _ in range(cfg.conv_blocks)))
        self.transformer = EncoderLayer(
            TransformerConfig(
                d_model=cfg.dim,
                num_heads=cfg.num_heads,
                ff_dim=cfg.ff_dim,
                dropout=cfg.attention_dropout,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        convolved = self.convs(x.transpose(1, 2)).transpose(1, 2)
        out: torch.Tensor = self.transformer(convolved)
        return out


class LateDropout(nn.Module):
    """Dropout that stays off (p = 0.0) until ``active`` is set; identity in eval mode."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p
        self.active = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (self.training and self.active and self.p > 0.0):
            return x
        return functional.dropout(x, self.p, training=True)


class HybridClassifier(SignClassifier):
    def __init__(self, num_classes: int, cfg: HybridConfig, features: StandardisedFeatures) -> None:
        super().__init__(features)
        self.late_dropout_start = cfg.late_dropout_start
        self.proj = nn.Linear(self.features.output_dim, cfg.dim)
        self.stages = nn.Sequential(*(Stage(cfg) for _ in range(cfg.num_stages)))
        self.head_dropout = LateDropout(cfg.head_dropout)
        self.head = nn.Linear(cfg.dim, num_classes)

    def set_training_progress(self, epoch: int, epochs: int) -> None:
        """Head dropout is active from the first epoch after ``late_dropout_start`` of all
        epochs, e.g. epochs 76-150 of 150 for the default 0.5."""
        self.head_dropout.active = (epoch - 1) >= self.late_dropout_start * epochs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.stages(self.proj(self.features(x)))
        logits: torch.Tensor = self.head(self.head_dropout(hidden.mean(dim=1)))
        return logits

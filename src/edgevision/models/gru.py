"""Bidirectional GRU over the frame sequence, pooled by mean and max over time."""

import torch
from torch import nn

from edgevision.models.config import GRUConfig
from edgevision.models.features import SignClassifier, StandardisedFeatures


class GRUClassifier(SignClassifier):
    def __init__(
        self, num_classes: int, cfg: GRUConfig, features: StandardisedFeatures | None = None
    ) -> None:
        super().__init__(features)
        self.proj = nn.Sequential(
            nn.Linear(self.features.output_dim, cfg.proj_dim),
            nn.LayerNorm(cfg.proj_dim),
            nn.GELU(),
            nn.Dropout(cfg.input_dropout),
        )
        self.gru = nn.GRU(
            cfg.proj_dim,
            cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.rnn_dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(cfg.head_dropout), nn.Linear(4 * cfg.hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.gru(self.proj(self.features(x)))
        pooled = torch.cat([sequence.mean(dim=1), sequence.amax(dim=1)], dim=-1)
        logits: torch.Tensor = self.head(pooled)
        return logits

"""Small pre-norm transformer encoder with hand-written attention.

nn.MultiheadAttention and nn.TransformerEncoderLayer are avoided on purpose: their fused fast
paths complicate export, and plain Linear + MatMul ops are what INT8 quantisation works on.
"""

import torch
from torch import nn

from edgevision.models.config import TransformerConfig
from edgevision.models.features import SignClassifier, StandardisedFeatures


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, H, T, D / H)."""
        return x.reshape(-1, x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self._heads(self.query(x)) * self.scale
        key = self._heads(self.key(x))
        value = self._heads(self.value(x))
        weights = torch.softmax(torch.matmul(query, key.transpose(-2, -1)), dim=-1)
        context = torch.matmul(self.dropout(weights), value)
        out: torch.Tensor = self.out(context.transpose(1, 2).flatten(2))
        return out


class EncoderLayer(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(cfg.d_model)
        self.attention = SelfAttention(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.feed_forward_norm = nn.LayerNorm(cfg.d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ff_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_dim, cfg.d_model),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.attention_norm(x)))
        out: torch.Tensor = x + self.dropout(self.feed_forward(self.feed_forward_norm(x)))
        return out


class TinyTransformer(SignClassifier):
    def __init__(
        self,
        num_classes: int,
        cfg: TransformerConfig,
        features: StandardisedFeatures | None = None,
    ) -> None:
        super().__init__(features)
        self.proj = nn.Linear(self.features.output_dim, cfg.d_model)
        frames = self.features.layout.num_frames
        self.position = nn.Parameter(torch.zeros(1, frames, cfg.d_model))
        nn.init.normal_(self.position, std=0.02)
        self.layers = nn.Sequential(*(EncoderLayer(cfg) for _ in range(cfg.num_layers)))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Sequential(nn.Dropout(cfg.head_dropout), nn.Linear(cfg.d_model, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.layers(self.proj(self.features(x)) + self.position)
        logits: torch.Tensor = self.head(self.norm(hidden).mean(dim=1))
        return logits

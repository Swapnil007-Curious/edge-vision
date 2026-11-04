"""Seed ensemble: N checkpoints of one experiment as one module that averages their logits,
exported to a single ONNX file.

The members share the input; each keeps its own feature block, because each seed's feature
statistics come from its own training run.
"""

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from edgevision.export.onnx_export import DEFAULT_OPSET, export_onnx
from edgevision.layout import InputLayout
from edgevision.models.features import SignClassifier
from edgevision.models.registry import load_checkpoint


class SeedEnsemble(nn.Module):
    """(B, T, frame_dim) -> (B, num_classes): the mean of the members' logits."""

    def __init__(self, members: Sequence[SignClassifier]) -> None:
        super().__init__()
        if len(members) < 2:
            raise ValueError("an ensemble needs at least two members")
        layouts = {member.features.layout for member in members}
        feature_sets = {member.features.feature_set for member in members}
        if len(layouts) != 1 or len(feature_sets) != 1:
            raise ValueError("ensemble members must share the input layout and feature set")
        self.members = nn.ModuleList(members)
        self.layout: InputLayout = members[0].features.layout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([member(x) for member in self.members], dim=0)
        return logits.mean(dim=0)


def load_ensemble(checkpoints: Sequence[Path]) -> tuple[SeedEnsemble, dict[int, str]]:
    """The ensemble of ``checkpoints`` (in eval mode) and their shared label map; refuses
    members of different architectures, class counts or label maps."""
    loaded = [load_checkpoint(path) for path in checkpoints]
    first = loaded[0]
    for path, item in zip(checkpoints, loaded, strict=True):
        if item.config != first.config or item.label_map != first.label_map:
            raise ValueError(f"{path} is not a seed of the same experiment as {checkpoints[0]}")
    ensemble = SeedEnsemble([item.model for item in loaded]).eval()
    return ensemble, first.label_map


def export_ensemble_onnx(
    checkpoints: Sequence[Path], path: Path, opset: int = DEFAULT_OPSET
) -> Path:
    """One ONNX file averaging the logits of ``checkpoints``, with input "landmarks"
    (batch, 32, frame_dim) and output "logits"."""
    ensemble, _ = load_ensemble(checkpoints)
    return export_onnx(ensemble, path, opset, layout=ensemble.layout)

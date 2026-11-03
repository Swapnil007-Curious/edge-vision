"""Evaluators turn (n, 32, 100) float32 inputs into logits, whatever runs the model.

final_eval.py talks only to this interface, so ONNX or quantised evaluators can be added
beside TorchCheckpointEvaluator without restructuring it.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from edgevision.models.registry import load_checkpoint

FloatArray = npt.NDArray[np.float32]


class Evaluator(ABC):
    """``name`` labels results; ``label_map`` maps class index to sign."""

    name: str
    num_classes: int
    label_map: dict[int, str]

    @abstractmethod
    def predict(self, X: FloatArray) -> FloatArray:
        """Return (n, num_classes) logits for (n, 32, 100) inputs laid out per edgevision.layout."""


class TorchCheckpointEvaluator(Evaluator):
    """A best.pt checkpoint run by PyTorch in float32."""

    name = "torch_fp32"

    def __init__(
        self, checkpoint: Path, device: torch.device | str = "cpu", batch_size: int = 1024
    ):
        self.device = torch.device(device)
        loaded = load_checkpoint(checkpoint, self.device)
        self.model = loaded.model
        self.num_classes = loaded.num_classes
        self.label_map = loaded.label_map
        self.batch_size = batch_size

    @torch.no_grad()
    def predict(self, X: FloatArray) -> FloatArray:
        parts = [np.empty((0, self.num_classes), dtype=np.float32)]
        for start in range(0, len(X), self.batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(X[start : start + self.batch_size]))
            logits = self.model(batch.to(self.device, dtype=torch.float32))
            parts.append(logits.float().cpu().numpy())
        return np.concatenate(parts)

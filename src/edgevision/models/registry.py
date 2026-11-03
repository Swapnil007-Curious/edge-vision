"""Model construction by name, parameter counts and checkpoint save/load."""

from pathlib import Path
from typing import Any, NamedTuple

import torch
from pydantic import TypeAdapter

from edgevision.layout import layout_dict
from edgevision.models.cnn import TemporalCNN
from edgevision.models.config import (
    CNNConfig,
    GRUConfig,
    ModelConfig,
    ModelName,
    TransformerConfig,
    default_model_config,
)
from edgevision.models.features import SignClassifier
from edgevision.models.gru import GRUClassifier
from edgevision.models.transformer import TinyTransformer

CHECKPOINT_FORMAT = 1
MIN_PARAMS, MAX_PARAMS = 300_000, 2_500_000

_MODEL_CONFIG: TypeAdapter[GRUConfig | CNNConfig | TransformerConfig] = TypeAdapter(ModelConfig)


def build_model(
    name: ModelName,
    num_classes: int,
    cfg: GRUConfig | CNNConfig | TransformerConfig | None = None,
) -> SignClassifier:
    """Build a model from its name; ``cfg`` defaults to that model's default hyperparameters."""
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}")
    config = default_model_config(name) if cfg is None else cfg
    if config.name != name:
        raise ValueError(f"model name {name!r} does not match config for {config.name!r}")
    if isinstance(config, GRUConfig):
        return GRUClassifier(num_classes, config)
    if isinstance(config, CNNConfig):
        return TemporalCNN(num_classes, config)
    return TinyTransformer(num_classes, config)


def count_parameters(model: torch.nn.Module) -> int:
    """Trainable parameters; buffers such as feature statistics are not counted."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_checkpoint(
    path: Path,
    model: SignClassifier,
    cfg: GRUConfig | CNNConfig | TransformerConfig,
    *,
    num_classes: int,
    label_map: dict[int, str],
) -> None:
    """Write through a temporary name so an interrupted save never leaves a truncated file."""
    payload = {
        "format": CHECKPOINT_FORMAT,
        "model_name": cfg.name,
        "model_config": cfg.model_dump(mode="json"),
        "num_classes": num_classes,
        "input_layout": layout_dict(),
        "label_map": {int(index): str(sign) for index, sign in label_map.items()},
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    temporary = path.with_name(f"{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


class LoadedCheckpoint(NamedTuple):
    model: SignClassifier
    config: GRUConfig | CNNConfig | TransformerConfig
    num_classes: int
    label_map: dict[int, str]


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> LoadedCheckpoint:
    """Rebuild the model in eval mode; refuses checkpoints made for another input layout."""
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: unsupported checkpoint format {payload.get('format')!r}")
    if payload["input_layout"] != layout_dict():
        raise ValueError(f"{path}: checkpoint input layout differs from edgevision.layout")
    config = _MODEL_CONFIG.validate_python(payload["model_config"])
    num_classes = int(payload["num_classes"])
    model = build_model(config.name, num_classes, config)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    label_map = {int(index): str(sign) for index, sign in payload["label_map"].items()}
    return LoadedCheckpoint(model, config, num_classes, label_map)

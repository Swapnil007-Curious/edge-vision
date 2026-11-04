"""Model construction by name, parameter counts and checkpoint save/load."""

from pathlib import Path
from typing import Any, NamedTuple

import torch
from pydantic import TypeAdapter

from edgevision.layout import V1_LAYOUT, InputLayout, layout_from_dict
from edgevision.models.cnn import TemporalCNN
from edgevision.models.config import (
    AnyModelConfig,
    CNNConfig,
    GRUConfig,
    HybridConfig,
    ModelConfig,
    ModelName,
    default_model_config,
)
from edgevision.models.features import (
    FEATURE_SETS,
    FeatureBlock,
    FeatureSet,
    SignClassifier,
    StandardisedFeatures,
)
from edgevision.models.features_v2 import FeatureBlockV2
from edgevision.models.gru import GRUClassifier
from edgevision.models.hybrid import HybridClassifier
from edgevision.models.transformer import TinyTransformer

CHECKPOINT_FORMAT = 1
# Ranges for 250 classes: the v1 architectures, and the hybrid.
MIN_PARAMS, MAX_PARAMS = 300_000, 2_500_000
HYBRID_MIN_PARAMS, HYBRID_MAX_PARAMS = 1_500_000, 6_000_000

_MODEL_CONFIG: TypeAdapter[AnyModelConfig] = TypeAdapter(ModelConfig)


def build_feature_block(
    feature_set: FeatureSet = "v1", layout: InputLayout = V1_LAYOUT
) -> StandardisedFeatures:
    if feature_set == "v1":
        return FeatureBlock(layout)
    if feature_set == "v2":
        return FeatureBlockV2(layout)
    raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got {feature_set!r}")


def build_model(
    name: ModelName,
    num_classes: int,
    cfg: AnyModelConfig | None = None,
    *,
    feature_set: FeatureSet = "v1",
    layout: InputLayout = V1_LAYOUT,
) -> SignClassifier:
    """Build a model from its name; ``cfg`` defaults to that model's default hyperparameters.

    ``feature_set`` and ``layout`` choose the feature block; the defaults are v1.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}")
    config = default_model_config(name) if cfg is None else cfg
    if config.name != name:
        raise ValueError(f"model name {name!r} does not match config for {config.name!r}")
    features = build_feature_block(feature_set, layout)
    if isinstance(config, GRUConfig):
        return GRUClassifier(num_classes, config, features)
    if isinstance(config, CNNConfig):
        return TemporalCNN(num_classes, config, features)
    if isinstance(config, HybridConfig):
        return HybridClassifier(num_classes, config, features)
    return TinyTransformer(num_classes, config, features)


def count_parameters(model: torch.nn.Module) -> int:
    """Trainable parameters; buffers such as feature statistics are not counted."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_checkpoint(
    path: Path,
    model: SignClassifier,
    cfg: AnyModelConfig,
    *,
    num_classes: int,
    label_map: dict[int, str],
) -> None:
    """Write through a temporary name so an interrupted save never leaves a truncated file.

    A v1-feature checkpoint has no "feature_set" key, so its payload is as it was before v2.
    """
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "model_name": cfg.name,
        "model_config": cfg.model_dump(mode="json"),
        "num_classes": num_classes,
        "input_layout": model.features.layout.to_dict(),
        "label_map": {int(index): str(sign) for index, sign in label_map.items()},
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    if model.features.feature_set != "v1":
        payload["feature_set"] = model.features.feature_set
    temporary = path.with_name(f"{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def checkpoint_layout(path: Path) -> InputLayout:
    """The input layout a checkpoint was trained on, without building its model."""
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    try:
        return layout_from_dict(payload["input_layout"])
    except (KeyError, TypeError, ValueError) as error:
        message = f"{path}: checkpoint input layout differs from edgevision.layout"
        raise ValueError(message) from error


class LoadedCheckpoint(NamedTuple):
    model: SignClassifier
    config: AnyModelConfig
    num_classes: int
    label_map: dict[int, str]


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> LoadedCheckpoint:
    """Rebuild the model in eval mode; refuses an input layout description that is not
    edgevision.layout's description of its own pose selection."""
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: unsupported checkpoint format {payload.get('format')!r}")
    try:
        layout = layout_from_dict(payload["input_layout"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{path}: checkpoint input layout differs from edgevision.layout"
        ) from error
    config = _MODEL_CONFIG.validate_python(payload["model_config"])
    num_classes = int(payload["num_classes"])
    feature_set: FeatureSet = payload.get("feature_set", "v1")
    model = build_model(config.name, num_classes, config, feature_set=feature_set, layout=layout)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    label_map = {int(index): str(sign) for index, sign in payload["label_map"].items()}
    return LoadedCheckpoint(model, config, num_classes, label_map)

"""Training configuration, loaded from experiments/train_<model>.yaml."""

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import Field, model_validator

from edgevision.config import SplitConfig, _StrictModel
from edgevision.models.config import ModelConfig


class AugmentConfig(_StrictModel):
    """Per-sample probabilities of each transform, and their magnitudes."""

    affine_p: float = Field(default=0.8, ge=0.0, le=1.0)
    temporal_p: float = Field(default=0.5, ge=0.0, le=1.0)
    jitter_p: float = Field(default=0.5, ge=0.0, le=1.0)
    rotation_deg: float = Field(default=15.0, ge=0.0)
    scale_min: float = Field(default=0.85, gt=0.0)
    scale_max: float = Field(default=1.15, gt=0.0)
    translate: float = Field(default=0.1, ge=0.0)
    window_min: float = Field(default=0.8, gt=0.0, le=1.0)
    jitter_std: float = Field(default=0.01, ge=0.0)

    @model_validator(mode="after")
    def _check_scale(self) -> Self:
        if self.scale_min > self.scale_max:
            raise ValueError("scale_min must not exceed scale_max")
        return self

    @property
    def enabled(self) -> bool:
        return max(self.affine_p, self.temporal_p, self.jitter_p) > 0.0


class TrainConfig(_StrictModel):
    """AdamW with a linear warmup over ``warmup_fraction`` of all steps, then cosine decay."""

    model: ModelConfig
    processed_dir: Path
    split: SplitConfig = Field(default_factory=SplitConfig)
    epochs: int = Field(default=60, ge=1)
    batch_size: int = Field(default=256, ge=1)
    lr: float = Field(gt=0.0)
    weight_decay: float = Field(default=0.05, ge=0.0)
    warmup_fraction: float = Field(default=0.05, ge=0.0, lt=1.0)
    label_smoothing: float = Field(default=0.1, ge=0.0, lt=1.0)
    grad_clip: float = Field(default=1.0, gt=0.0)
    patience: int = Field(default=15, ge=1)
    augment: AugmentConfig = Field(default_factory=AugmentConfig)
    num_threads: int = Field(default=4, ge=1)  # CPU only.
    output_root: Path = Path("results/runs")


def load_train_config(
    path: Path,
    *,
    processed_dir: Path | None = None,
    output_root: Path | None = None,
    epochs: int | None = None,
    split_mode: Literal["signer_independent", "signer_dependent"] | None = None,
) -> TrainConfig:
    """Load a YAML training config; non-None keyword arguments override the YAML values."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")

    candidates: dict[str, Any] = {
        "processed_dir": processed_dir,
        "output_root": output_root,
        "epochs": epochs,
    }
    raw.update({key: value for key, value in candidates.items() if value is not None})
    if split_mode is not None:
        raw["split"] = {**(raw.get("split") or {}), "mode": split_mode}
    return TrainConfig.model_validate(raw)


def dump_train_config(cfg: TrainConfig) -> str:
    """YAML that ``load_train_config`` reads back to an equal config."""
    return yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)

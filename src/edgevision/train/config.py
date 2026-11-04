"""Training configuration, loaded from experiments/train_*.yaml.

Fields added for v2 (experiment, feature_set, recipe, ema_decay and the extra augmentations)
default to the v1 behaviour and are left out of a dumped config while they hold their defaults,
so a v1 run's config.yaml is unchanged.
"""

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import Field, model_validator

from edgevision.config import SplitConfig, _StrictModel
from edgevision.models.config import ModelConfig
from edgevision.models.features import FeatureSet

Recipe = Literal["v1", "v2"]


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
    # v2 additions; every probability defaults to 0.0, which leaves v1 batches unchanged.
    frame_dropout_p: float = Field(default=0.0, ge=0.0, le=1.0)
    frame_dropout_max: float = Field(default=0.2, ge=0.0, lt=1.0)
    time_mask_p: float = Field(default=0.0, ge=0.0, le=1.0)
    time_mask_windows: int = Field(default=2, ge=1)
    time_mask_max_frames: int = Field(default=4, ge=1)
    landmark_mask_p: float = Field(default=0.0, ge=0.0, le=1.0)
    landmark_mask_max: int = Field(default=4, ge=1)
    hand_rotation_p: float = Field(default=0.0, ge=0.0, le=1.0)
    hand_rotation_deg: float = Field(default=10.0, ge=0.0)

    @model_validator(mode="after")
    def _check_scale(self) -> Self:
        if self.scale_min > self.scale_max:
            raise ValueError("scale_min must not exceed scale_max")
        return self

    @property
    def v2_probabilities(self) -> tuple[float, float, float, float]:
        return self.frame_dropout_p, self.time_mask_p, self.landmark_mask_p, self.hand_rotation_p

    @property
    def enabled(self) -> bool:
        return max(self.affine_p, self.temporal_p, self.jitter_p, *self.v2_probabilities) > 0.0


V2_AUGMENT_FIELDS = (
    "frame_dropout_p",
    "frame_dropout_max",
    "time_mask_p",
    "time_mask_windows",
    "time_mask_max_frames",
    "landmark_mask_p",
    "landmark_mask_max",
    "hand_rotation_p",
    "hand_rotation_deg",
)
V2_TRAIN_FIELDS = ("experiment", "feature_set", "recipe", "ema_decay")


class TrainConfig(_StrictModel):
    """AdamW with a linear warmup over ``warmup_fraction`` of all steps, then cosine decay.

    ``experiment`` names the run directory (default "<model>_<split mode>", the v1 names);
    ``recipe`` is recorded in summary.json, and "v1" rules out EMA and the v2 augmentations.
    With ``ema_decay`` set, an exponential moving average of the weights is updated after every
    optimizer step and is what validation, checkpoint selection and best.pt use.
    """

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
    experiment: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    feature_set: FeatureSet = "v1"
    recipe: Recipe = "v1"
    ema_decay: float | None = Field(default=None, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _check_recipe(self) -> Self:
        if self.recipe == "v1" and (
            self.ema_decay is not None or max(self.augment.v2_probabilities) > 0.0
        ):
            raise ValueError("recipe v1 uses neither EMA nor the v2 augmentations")
        return self

    @property
    def experiment_name(self) -> str:
        return self.experiment or f"{self.model.name}_{self.split.mode}"


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
    """YAML that ``load_train_config`` reads back to an equal config. v2 fields that hold their
    defaults are left out, so a v1 config dumps exactly as it did before they existed."""
    payload = cfg.model_dump(mode="json")
    for name in V2_TRAIN_FIELDS:
        if getattr(cfg, name) == TrainConfig.model_fields[name].default:
            del payload[name]
    for name in V2_AUGMENT_FIELDS:
        if getattr(cfg.augment, name) == AugmentConfig.model_fields[name].default:
            del payload["augment"][name]
    return yaml.safe_dump(payload, sort_keys=False)

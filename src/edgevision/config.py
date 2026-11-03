"""Pydantic configuration models, loaded from YAML."""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edgevision.data.landmarks import DEFAULT_POSE_INDICES, validate_pose_indices


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PreprocessConfig(_StrictModel):
    data_root: Path
    output_dir: Path
    num_frames: int = Field(default=32, ge=1)
    use_z: bool = False  # z from this source is unreliable; kept optional.
    pose_indices: list[int] = Field(default_factory=lambda: list(DEFAULT_POSE_INDICES))
    min_hand_frames: int = Field(default=3, ge=1)
    limit: int | None = Field(default=None, ge=1)
    num_workers: int = Field(default=4, ge=1)
    seed: int = Field(default=42, ge=0)

    @field_validator("pose_indices")
    @classmethod
    def _check_pose_indices(cls, value: list[int]) -> list[int]:
        validate_pose_indices(value)
        return value


class SplitConfig(_StrictModel):
    mode: Literal["signer_independent", "signer_dependent"] = "signer_independent"
    n_test_signers: int = Field(default=4, ge=1)
    n_val_signers: int = Field(default=3, ge=1)
    val_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    test_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    seed: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def _check_fractions(self) -> Self:
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError("val_fraction + test_fraction must be below 1.0")
        return self


class ExperimentConfig(_StrictModel):
    preprocess: PreprocessConfig
    split: SplitConfig = Field(default_factory=SplitConfig)


def load_config(
    path: Path,
    *,
    data_root: Path | None = None,
    output_dir: Path | None = None,
    limit: int | None = None,
) -> ExperimentConfig:
    """Load a YAML experiment config; non-None keyword arguments override the YAML values."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")

    candidates = {"data_root": data_root, "output_dir": output_dir, "limit": limit}
    overrides = {key: value for key, value in candidates.items() if value is not None}
    raw["preprocess"] = {**(raw.get("preprocess") or {}), **overrides}
    return ExperimentConfig.model_validate(raw)

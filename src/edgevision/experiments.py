"""Experiment registry: the v1 grid and the v2 ablation grid.

The v2 grid isolates one change per step against the v1 CNN (cnn_signer_independent):
  b  v1 -> v2 features             (cnn, v2 features, v1 recipe)
  c  v1 -> v2 recipe               (cnn, v2 features, v2 recipe)
  d  cnn -> hybrid architecture    (hybrid, v2 features, v2 recipe)
Every experiment is signer-independent with the v1 split settings.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from edgevision.models.config import MODEL_NAMES, ModelName
from edgevision.models.features import FeatureSet
from edgevision.runs import PRIMARY_SPLIT_MODE, run_name
from edgevision.train.config import Recipe

GridName = str
GRIDS: tuple[GridName, ...] = ("v1", "v2")


@dataclass(frozen=True)
class Experiment:
    name: str
    config: str  # File name under experiments/.
    model: ModelName
    feature_set: FeatureSet
    recipe: Recipe
    split_mode: str = PRIMARY_SPLIT_MODE


V1_EXPERIMENTS: tuple[Experiment, ...] = tuple(
    Experiment(run_name(model, mode), f"train_{model}.yaml", model, "v1", "v1", mode)
    for mode in (PRIMARY_SPLIT_MODE, "signer_dependent")
    for model in MODEL_NAMES
)
V2_EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment("b_cnn_v2feat_v1recipe", "train_b_cnn_v2feat_v1recipe.yaml", "cnn", "v2", "v1"),
    Experiment("c_cnn_v2feat_v2recipe", "train_c_cnn_v2feat_v2recipe.yaml", "cnn", "v2", "v2"),
    Experiment(
        "d_hybrid_v2feat_v2recipe", "train_d_hybrid_v2feat_v2recipe.yaml", "hybrid", "v2", "v2"
    ),
)
EXPERIMENTS: dict[str, Experiment] = {item.name: item for item in V1_EXPERIMENTS + V2_EXPERIMENTS}


def v2_grid_plan(seeds: Sequence[int]) -> list[tuple[Experiment, int]]:
    """Every v2 experiment at every seed, seed-major: all three finish seed 0 first."""
    if not seeds:
        raise ValueError("at least one seed is required")
    return [(experiment, seed) for seed in seeds for experiment in V2_EXPERIMENTS]

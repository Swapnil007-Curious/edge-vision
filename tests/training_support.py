"""Synthetic preprocessing output and script loading for the training test suite.

The builder writes what scripts/preprocess.py writes (features.npz, manifest.csv and optionally
splits_signer_independent.json) without running preprocessing. Each class follows its own hand
trajectory, so models can learn it; some points and frames are missing (0.0).
"""

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
import pandas as pd

from conftest import PARTICIPANTS, SIGNS
from edgevision.config import SplitConfig
from edgevision.data.io import LABEL_DTYPE, MANIFEST_COLUMNS, PARTICIPANT_DTYPE, SEQUENCE_DTYPE
from edgevision.data.landmarks import LEFT_HAND, RIGHT_HAND
from edgevision.data.splits import make_splits, splits_to_json, summarize_splits
from edgevision.layout import NUM_FRAMES, NUM_LANDMARKS

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
EXPERIMENTS = REPO_ROOT / "experiments"
N_EXCLUDED = 3
# Pose slots in layout order: nose, shoulders (11, 12), elbows (13, 14), wrists (15, 16).
POSE_ANCHORS = np.array(
    [[0.0, -1.0], [0.5, 0.0], [-0.5, 0.0], [0.7, 0.6], [-0.7, 0.6], [0.4, 0.9], [-0.4, 0.9]]
)
LEFT_ANCHOR = np.array([0.5, 0.5])


def load_script(name: str) -> ModuleType:
    """Import scripts/<name>.py, which is not part of the installed package."""
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def trajectory(label: int, t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """(T, 2) dominant-hand path for a class: circle, sweeps, shake, zigzag."""
    if label == 0:
        return 0.5 * np.stack([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)], axis=1)
    if label == 1:
        return np.stack([1.2 * t - 0.6, np.zeros_like(t)], axis=1)
    if label == 2:
        return np.stack([np.zeros_like(t), 1.2 * t - 0.6], axis=1)
    if label == 3:
        return np.stack([0.15 * np.sin(6 * np.pi * t), 0.3 * np.ones_like(t)], axis=1)
    triangle = 2.0 * np.abs(2.0 * (2.0 * t % 1.0) - 1.0) - 1.0
    return 0.4 * np.stack([triangle, triangle], axis=1)


@dataclass(frozen=True)
class SyntheticProcessed:
    root: Path
    manifest: pd.DataFrame
    features: dict[str, npt.NDArray[np.generic]]


def build_processed_dir(
    root: Path,
    *,
    per_cell: int = 8,
    seed: int = 0,
    write_splits: bool = True,
    extreme_eval: bool = False,
) -> SyntheticProcessed:
    """8 participants x 5 classes x ``per_cell`` sequences, rows shuffled, plus excluded rows.

    ``extreme_eval`` scales every val and test coordinate to around 1000, to show that feature
    statistics ignore those splits.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, NUM_FRAMES)
    hand_shape = rng.normal(0.0, 0.08, (21, 2))
    records, coords, presences = [], [], []
    ordinal = 0
    for participant in PARTICIPANTS:
        offset = rng.uniform(-0.15, 0.15, 2)
        scale = rng.uniform(0.9, 1.1)
        for label, sign in enumerate(SIGNS):
            for _ in range(per_cell):
                points = np.zeros((NUM_FRAMES, NUM_LANDMARKS, 2))
                path = scale * trajectory(label, t) + offset
                points[:, RIGHT_HAND] = path[:, None] + hand_shape + rng.normal(0, 0.03, (21, 2))
                points[:, 42:] = POSE_ANCHORS + rng.normal(0.0, 0.02, (NUM_FRAMES, 7, 2))
                presence = np.ones((NUM_FRAMES, 2), dtype=np.uint8)
                if rng.random() < 0.4:
                    points[:, LEFT_HAND] = LEFT_ANCHOR + hand_shape + rng.normal(0, 0.03, (21, 2))
                else:
                    presence[:, 0] = 0  # Left hand absent: its points stay 0.0.
                dropped = rng.choice(NUM_FRAMES, size=rng.integers(0, 4), replace=False)
                points[dropped, RIGHT_HAND] = 0.0
                presence[dropped, 1] = 0
                points[rng.random(NUM_FRAMES) < 0.05, 42] = 0.0  # The nose goes missing too.

                coords.append(points)
                presences.append(presence)
                records.append(
                    {
                        "sequence_id": 1_000_000_000 + ordinal * 40_000_019,
                        "participant_id": participant,
                        "sign": sign,
                        "label": label,
                    }
                )
                ordinal += 1

    kept = pd.DataFrame(records).assign(
        excluded=False, reason="", t_raw=40, t_kept=40, mirrored=False, norm_method="shoulders"
    )
    excluded = kept.iloc[:N_EXCLUDED].assign(
        sequence_id=lambda frame: frame["sequence_id"] + 7,
        excluded=True,
        reason="insufficient_hand_frames",
        t_kept=0,
        norm_method="",
    )
    manifest = pd.concat([kept, excluded], ignore_index=True)
    manifest = manifest.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    manifest = manifest.loc[:, list(MANIFEST_COLUMNS)]

    order = rng.permutation(len(kept))
    features = {
        "X": np.stack(coords)[order].astype(np.float16),
        "presence": np.stack(presences)[order],
        "y": kept["label"].to_numpy(dtype=LABEL_DTYPE)[order],
        "participant_id": kept["participant_id"].to_numpy(dtype=PARTICIPANT_DTYPE)[order],
        "sequence_id": kept["sequence_id"].to_numpy(dtype=SEQUENCE_DTYPE)[order],
    }

    root.mkdir(parents=True, exist_ok=True)
    cfg = SplitConfig()
    splits = make_splits(manifest, cfg)
    if write_splits:
        payload = splits_to_json(splits, summarize_splits(splits, manifest), cfg)
        (root / "splits_signer_independent.json").write_text(json.dumps(payload), encoding="utf-8")
    if extreme_eval:
        held_out = np.isin(features["sequence_id"], np.concatenate([splits["val"], splits["test"]]))
        X = features["X"]
        X[held_out] = np.where(X[held_out] != 0, np.float16(1000.0), np.float16(0.0))

    np.savez_compressed(root / "features.npz", allow_pickle=False, **features)
    manifest.to_csv(root / "manifest.csv", index=False)
    return SyntheticProcessed(root, manifest, features)


def snapshot(directory: Path) -> dict[str, tuple[int, int]]:
    """Relative path -> (size, mtime_ns) for every file, to prove a directory was not written."""
    return {
        path.relative_to(directory).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }

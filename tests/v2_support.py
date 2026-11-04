"""Synthetic v2 preprocessing output (pose 0-16, L = 59) for the v2 test suite.

Like training_support.build_processed_dir, but with the v2 landmark selection and a minimal
preprocess_report.json that records it, as scripts/preprocess.py does.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from conftest import PARTICIPANTS, SIGNS
from edgevision.config import PreprocessConfig, SplitConfig
from edgevision.data.io import (
    LABEL_DTYPE,
    MANIFEST_COLUMNS,
    PARTICIPANT_DTYPE,
    SEQUENCE_DTYPE,
    landmark_selection,
)
from edgevision.data.landmarks import LEFT_HAND, RIGHT_HAND, pose_slot
from edgevision.data.splits import make_splits, splits_to_json, summarize_splits
from edgevision.layout import NUM_FRAMES, InputLayout
from training_support import N_EXCLUDED, trajectory

V2_POSE_INDICES = tuple(range(17))
V2_LAYOUT = InputLayout.for_pose_indices(V2_POSE_INDICES)
# Pose 0-16 in normalised units: nose, eyes (inner, centre, outer; left then right), ears,
# mouth corners, shoulders, elbows, wrists.
V2_POSE_ANCHORS = np.array(
    [
        [0.0, -1.0],
        [0.08, -1.1],
        [0.14, -1.1],
        [0.2, -1.1],
        [-0.08, -1.1],
        [-0.14, -1.1],
        [-0.2, -1.1],
        [0.3, -1.0],
        [-0.3, -1.0],
        [0.07, -0.85],
        [-0.07, -0.85],
        [0.5, 0.0],
        [-0.5, 0.0],
        [0.7, 0.6],
        [-0.7, 0.6],
        [0.4, 0.9],
        [-0.4, 0.9],
    ]
)
LEFT_ANCHOR = np.array([0.5, 0.5])


@dataclass(frozen=True)
class SyntheticV2:
    root: Path
    manifest: pd.DataFrame
    features: dict[str, npt.NDArray[np.generic]]


def build_v2_processed_dir(
    root: Path,
    *,
    per_cell: int = 8,
    seed: int = 0,
    split: SplitConfig | None = None,
    write_splits: bool = True,
    extreme_eval: bool = False,
) -> SyntheticV2:
    """8 participants x 5 classes x ``per_cell`` sequences on the v2 layout, plus excluded rows.

    Mouth corners and the nose go missing in some frames, so the v2 missing-point rule is
    exercised. ``extreme_eval`` scales val and test coordinates to around 1000.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, NUM_FRAMES)
    hand_shape = rng.normal(0.0, 0.08, (21, 2))
    landmarks = V2_LAYOUT.num_landmarks
    first_pose = pose_slot(V2_POSE_INDICES, 0)
    mouth = [pose_slot(V2_POSE_INDICES, 9), pose_slot(V2_POSE_INDICES, 10)]
    records, coords, presences = [], [], []
    ordinal = 0
    for participant in PARTICIPANTS:
        offset = rng.uniform(-0.15, 0.15, 2)
        scale = rng.uniform(0.9, 1.1)
        for label, sign in enumerate(SIGNS):
            for _ in range(per_cell):
                points = np.zeros((NUM_FRAMES, landmarks, 2))
                path = scale * trajectory(label, t) + offset
                points[:, RIGHT_HAND] = path[:, None] + hand_shape + rng.normal(0, 0.03, (21, 2))
                pose_noise = rng.normal(0.0, 0.02, (NUM_FRAMES, len(V2_POSE_INDICES), 2))
                points[:, first_pose:] = V2_POSE_ANCHORS + pose_noise
                presence = np.ones((NUM_FRAMES, 2), dtype=np.uint8)
                if rng.random() < 0.4:
                    left = LEFT_ANCHOR + hand_shape + rng.normal(0, 0.03, (NUM_FRAMES, 21, 2))
                    points[:, LEFT_HAND] = left
                else:
                    presence[:, 0] = 0
                dropped = rng.choice(NUM_FRAMES, size=rng.integers(0, 4), replace=False)
                points[dropped, RIGHT_HAND] = 0.0
                presence[dropped, 1] = 0
                points[rng.random(NUM_FRAMES) < 0.05, first_pose] = 0.0
                points[rng.random(NUM_FRAMES) < 0.05, mouth[0]] = 0.0

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
    cfg = split or SplitConfig()
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
    preprocess = PreprocessConfig(
        data_root=Path("unused"), output_dir=root, pose_indices=list(V2_POSE_INDICES)
    )
    report = {
        "config": {"preprocess": preprocess.model_dump(mode="json")},
        "landmarks": landmark_selection(preprocess),
    }
    (root / "preprocess_report.json").write_text(json.dumps(report), encoding="utf-8")
    return SyntheticV2(root, manifest, features)

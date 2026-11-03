"""Synthetic asl-signs dataset builder shared by the test suite."""

import importlib.util
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from edgevision.data.landmarks import LANDMARK_COUNTS

FloatArray = npt.NDArray[np.float64]

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preprocess.py"

# Some participant ids exceed int16 and sequence ids exceed int32, so the storage dtypes are
# exercised on values that would wrap if they were too narrow.
PARTICIPANTS = (2044, 4718, 16069, 26734, 37055, 49445, 55372, 62590)
# "null" must survive CSV parsing as a string.
SIGNS = ("apple", "bird", "cat", "null", "zebra")
CASE_CYCLE = ("both_hands", "right_hand_only", "left_hand_only", "partial_hands", "pose_missing")
# Sequence ordinals whose case replaces the cycled one in the default dataset.
SPECIAL_CASES: Mapping[int, str] = {
    7: "no_hands",
    47: "no_hands",
    87: "no_hands",
    19: "coincident_shoulders",
    33: "corrupted",
}


def load_cli_module() -> ModuleType:
    """Import scripts/preprocess.py, which is not part of the installed package."""
    spec = importlib.util.spec_from_file_location("cli_preprocess", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def landmark_blocks(
    n_frames: int, rng: np.random.Generator, case: str = "both_hands"
) -> dict[str, FloatArray]:
    """Per-type (T, n, 3) arrays with plausible positions, modified according to ``case``."""
    phase = np.linspace(0.0, 2.0 * np.pi, n_frames)[:, None, None]

    def block(n_points: int, center: tuple[float, float, float]) -> FloatArray:
        anchor = np.asarray(center) + rng.uniform(-0.05, 0.05, (n_points, 3))
        return anchor[None] + 0.01 * np.sin(phase)

    blocks = {
        "face": block(LANDMARK_COUNTS["face"], (0.5, 0.25, 0.0)),
        "left_hand": block(LANDMARK_COUNTS["left_hand"], (0.65, 0.45, -0.02)),
        "pose": block(LANDMARK_COUNTS["pose"], (0.5, 0.6, 0.0)),
        "right_hand": block(LANDMARK_COUNTS["right_hand"], (0.35, 0.45, -0.02)),
    }
    blocks["pose"][:, 11, :2] = (0.65, 0.55)
    blocks["pose"][:, 12, :2] = (0.35, 0.55)

    if case == "right_hand_only":
        blocks["left_hand"][:] = np.nan
    elif case == "left_hand_only":
        blocks["right_hand"][:] = np.nan
    elif case == "partial_hands":
        n_missing = n_frames // 3
        blocks["left_hand"][:] = np.nan
        blocks["right_hand"][:n_missing] = np.nan
    elif case == "no_hands":
        blocks["left_hand"][:] = np.nan
        blocks["right_hand"][:] = np.nan
    elif case == "pose_missing":
        blocks["pose"][:] = np.nan
    elif case == "coincident_shoulders":
        blocks["pose"][:, 12, :2] = blocks["pose"][:, 11, :2]
    return blocks


def long_frame(blocks: Mapping[str, FloatArray], *, first_frame: int = 20) -> pd.DataFrame:
    """Long-format frame sorted by frame; types absent from ``blocks`` have no rows at all."""
    n_frames = next(iter(blocks.values())).shape[0]
    frames = np.arange(first_frame, first_frame + n_frames, dtype=np.int16)

    parts = []
    for kind, block in blocks.items():
        n_points = block.shape[1]
        part = pd.DataFrame(
            {
                "frame": np.repeat(frames, n_points),
                "type": kind,
                "landmark_index": np.tile(np.arange(n_points, dtype=np.int16), n_frames),
                "x": block[..., 0].ravel(),
                "y": block[..., 1].ravel(),
                "z": block[..., 2].ravel(),
            }
        )
        parts.append(part)

    df = pd.concat(parts, ignore_index=True).sort_values("frame", kind="stable")
    row_ids = zip(
        df["frame"].tolist(), df["type"].tolist(), df["landmark_index"].tolist(), strict=True
    )
    df.insert(1, "row_id", [f"{frame}-{kind}-{index}" for frame, kind, index in row_ids])
    return df.reset_index(drop=True)


@dataclass(frozen=True)
class FakeDataset:
    root: Path
    table: pd.DataFrame  # train.csv rows plus the "case" column.


def build_fake_dataset(
    root: Path,
    *,
    participants: tuple[int, ...] = PARTICIPANTS,
    signs: tuple[str, ...] = SIGNS,
    sequences_per_cell: int = 3,
    case_cycle: tuple[str, ...] = CASE_CYCLE,
    special_cases: Mapping[int, str] = SPECIAL_CASES,
    seed: int = 0,
) -> FakeDataset:
    """Write train.csv, the sign map and one parquet per sequence under ``root``."""
    rng = np.random.default_rng(seed)
    rows = []
    ordinal = 0
    for participant in participants:
        for sign in signs:
            for _ in range(sequences_per_cell):
                sequence_id = 1_000_000_000 + ordinal * 40_000_019
                case = special_cases.get(ordinal, case_cycle[ordinal % len(case_cycle)])
                relative = f"train_landmark_files/{participant}/{sequence_id}.parquet"
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if case == "corrupted":
                    target.write_bytes(b"this is not a parquet file")
                else:
                    n_frames = int(rng.integers(10, 61))
                    blocks = landmark_blocks(n_frames, rng, case)
                    long_frame(blocks).to_parquet(target, engine="pyarrow", index=False)
                rows.append(
                    {
                        "path": relative,
                        "participant_id": participant,
                        "sequence_id": sequence_id,
                        "sign": sign,
                        "case": case,
                    }
                )
                ordinal += 1

    table = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    table.drop(columns="case").to_csv(root / "train.csv", index=False)
    sign_map = {sign: index for index, sign in enumerate(signs)}
    (root / "sign_to_prediction_index_map.json").write_text(json.dumps(sign_map), encoding="utf-8")
    return FakeDataset(root, table)


@pytest.fixture(scope="session")
def fake_dataset(tmp_path_factory: pytest.TempPathFactory) -> FakeDataset:
    return build_fake_dataset(tmp_path_factory.mktemp("asl-signs"))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
